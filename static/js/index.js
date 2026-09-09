window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

    var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    document.querySelectorAll('video.gallery-video').forEach(function(video, index) {
        video.setAttribute('aria-label', 'GRN gallery video ' + (index + 1));
        if (!reduceMotion) {
            video.autoplay = true;
            var playback = video.play();
            if (playback) {
                playback.catch(function() {});
            }
        }
    });

})
