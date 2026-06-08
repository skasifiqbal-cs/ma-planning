(define (problem strips-sat-x-1) (:domain satellite)
(:objects
	satellite2 - satellite
	satellite1 - satellite
	satellite0 - satellite
	star10 - direction
	thermograph2 - mode
	star6 - direction
	groundstation3 - direction
	instrument2 - instrument
	instrument3 - instrument
	instrument0 - instrument
	instrument1 - instrument
	instrument4 - instrument
	planet5 - direction
	planet4 - direction
	phenomenon8 - direction
	infrared1 - mode
	infrared3 - mode
	star7 - direction
	spectrograph0 - mode
	star1 - direction
	star0 - direction
	star2 - direction
	star9 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 star1)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 phenomenon8)
	(supports instrument1 infrared3)
	(calibration_target instrument1 star2)
	(supports instrument2 infrared1)
	(supports instrument2 infrared3)
	(supports instrument2 thermograph2)
	(calibration_target instrument2 star2)
	(supports instrument3 infrared1)
	(supports instrument3 infrared3)
	(supports instrument3 spectrograph0)
	(calibration_target instrument3 star2)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 star6)
	(supports instrument4 infrared3)
	(calibration_target instrument4 star0)
	(on_board instrument4 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star6)
)
(:goal
	(and
		(have_image planet4 thermograph2)
		(have_image planet5 spectrograph0)
		(have_image star6 thermograph2)
		(have_image star7 infrared3)
		(have_image phenomenon8 spectrograph0)
		(have_image star9 infrared1)
		(have_image star10 infrared3)
	)
)
)