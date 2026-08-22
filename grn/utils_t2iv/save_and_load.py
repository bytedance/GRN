import gc
import os
import os.path as osp
import subprocess
import time
import re
from typing import List, Optional, Tuple

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import glob
import shutil
from grn.utils_t2iv import arg_util
import grn.utils_t2iv.dist as dist

def glob_with_global_step(pattern, recursive=False): 
    def extract_ep_iter(filename):
        match = re.search(r'global_step_(\d+)', filename)
        if match:
            iter_idx = int(match.group(1))
            return iter_idx
        return 0
    return sorted(glob.glob(pattern, recursive=recursive), key=lambda x: extract_ep_iter(os.path.basename(x)), reverse=True)
        

class CKPTSaver(object):
    def __init__(self, is_master: bool, eval_milestone: List[Tuple[float, float]]):
        self.is_master = is_master
        self.time_stamp = torch.tensor([time.time() - 1e5, time.time()], device=dist.get_device())
        self.sp_also: subprocess.Popen = None
        self.sp_best: subprocess.Popen = None
        self.sp_backup: subprocess.Popen = None
        self.acc_str, self.eval_milestone = '[no acc str]', eval_milestone
    
    def sav(
        self, args: arg_util.Args, g_it: int, next_ep: int, next_it: int, trainer,
        acc_str: Optional[str] = None, eval_milestone: Optional[List[Tuple[float, float]]] = None,
        also_save_to: str = None, best_save_to: str = None,
    ):
        
        if acc_str is not None: self.acc_str = acc_str
        if eval_milestone is not None: self.eval_milestone = eval_milestone
        
        fname = f'global_step_{g_it}.pth'
        local_out_ckpt = os.path.join(args.local_out_path, fname)
        
        # NOTE: all rank should call this state_dict(), not master only!
        trainer_state = trainer.state_dict()
        
        if self.is_master:
            stt = time.time()
            torch.save({
                'args':         args.state_dict(),
                'gpt_training': args.gpt_training,
                'arch':         args.model,
                'epoch':        next_ep,
                'iter':         next_it,
                'trainer':      trainer_state,
                'g_it':         g_it,
            }, local_out_ckpt)
            
            print(f'[CKPTSaver][rank00] dbg: {args.bed=}', flush=True)                
            def auto_sync(source_filename, target_filename):
                print(f'[CKPTSaver] auto_save {source_filename} -> {target_filename}', flush=True)
                def _sync_worker():
                    try:
                        import shutil
                        from grn.utils.safe_rm import safe_remove
                        if os.path.isdir(source_filename):
                            shutil.copytree(source_filename, target_filename, dirs_exist_ok=True)
                        else:
                            shutil.copy2(source_filename, target_filename)
                        if source_filename.endswith('.pth') and (osp.abspath(source_filename) != osp.abspath(target_filename)):
                            safe_remove(source_filename, osp.dirname(source_filename))
                    except Exception as e:
                        print(f'[CKPTSaver] auto_save failed: {e}', flush=True)
                
                import threading
                self.sp_backup = threading.Thread(target=_sync_worker)
                self.sp_backup.start()

            local_files = glob.glob(f"{args.local_out_path}/*")
            for filename in local_files:
                basename = os.path.basename(filename)
                target_filename = f'{args.bed}/{basename}'
                if basename.endswith('.pth'):
                    if not os.path.isfile(target_filename):
                        auto_sync(filename, target_filename)
                else:
                    auto_sync(filename, target_filename)                    
            cost = time.time() - stt
            print(f'[CKPTSaver][rank00] cost: {cost:.2f}s', flush=True)
        
        del trainer_state
        time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
        dist.barrier()

def auto_resume(args: arg_util.Args, pattern='ckpt*.pth') -> Tuple[List[str], int, int, str, List[Tuple[float, float]], dict, dict]:
    info = []
    resume = ''
    if args.auto_resume:
        for dd in (args.local_out_path, args.bed):
            all_ckpt = glob_with_global_step(os.path.join(dd, pattern))
            if len(all_ckpt): break
        if len(all_ckpt) == 0:
            info.append(f'[auto_resume] no ckpt found @ {pattern}')
            info.append(f'[auto_resume quit]')
        else:
            resume = all_ckpt[0]
            info.append(f'[auto_resume] auto load from @ {resume} ...')
    else:
        info.append(f'[auto_resume] disabled')
        info.append(f'[auto_resume quit]')
    
    if len(resume) == 0:
        return info, 0, 0, '[no acc str]', [], {}, {}

    print(f'auto resume from {resume}')

    try:
        tgt_file = os.path.join(args.local_out_path, osp.basename(resume))
        os.makedirs(osp.dirname(tgt_file), exist_ok=True)
        if osp.abspath(resume) != osp.abspath(tgt_file):
            print(f'[load model] copy {resume} to {tgt_file}')
            shutil.copyfile(resume, tgt_file)
        ckpt = torch.load(tgt_file, map_location='cpu')
    except Exception as e:
        info.append(f'[auto_resume] failed, {e} @ {resume}')
        if len(all_ckpt) < 2:
            return info, 0, 0, '[no acc str]', [], {}, {}
        try: # another chance to load from bytenas
            ckpt = torch.load(all_ckpt[1], map_location='cpu')
        except Exception as e:
            info.append(f'[auto_resume] failed, {e} @ {all_ckpt[1]}')
            return info, 0, 0, '[no acc str]', [], {}, {}
    
    dist.barrier()
    ep, it, g_it = ckpt['epoch'], ckpt['iter'], ckpt.get('g_it', 0)
    eval_milestone = ckpt.get('milestones', [])
    info.append(f'[auto_resume success] resume from ep{ep}, it{it},    eval_milestone: {eval_milestone}')
    return info, ep, it, ckpt.get('acc_str', '[no acc str]'), eval_milestone, ckpt['trainer'], ckpt['args']

def _is_trace_training_duration():
    if (
        metrics_client is not None
        and os.environ.get('ENABLE_TRAINING_DURATION_METRICS_COLLECTION', 'false') == 'true'
        and os.environ.get('MERLIN_JOB_ID', None) is not None
        and os.environ.get("ARNOLD_ROBUST_TRAINING", None) == '1'
    ):
        # collect metrics from any of the following executor
        executor_list = ["executor-0-0", "executor-0-1", "executor-0-2",
                         "executor-0", "executor-1", "executor-2"]
        for ending in executor_list:
            if os.environ.get("MY_POD_NAME", "").endswith(ending):
                return True
    return False
