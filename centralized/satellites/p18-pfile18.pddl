(define (problem strips-sat-x-1) (:domain satellite)
(:objects
	planet11 - direction
	thermograph2 - mode
	planet14 - direction
	planet19 - direction
	instrument9 - instrument
	satellite3 - satellite
	satellite2 - satellite
	satellite1 - satellite
	satellite0 - satellite
	satellite4 - satellite
	star15 - direction
	star10 - direction
	thermograph0 - mode
	thermograph3 - mode
	phenomenon13 - direction
	thermograph4 - mode
	phenomenon16 - direction
	phenomenon17 - direction
	instrument2 - instrument
	phenomenon12 - direction
	groundstation2 - direction
	phenomenon18 - direction
	instrument8 - instrument
	instrument1 - instrument
	planet20 - direction
	instrument3 - instrument
	instrument0 - instrument
	planet23 - direction
	instrument6 - instrument
	instrument7 - instrument
	instrument4 - instrument
	instrument5 - instrument
	planet7 - direction
	phenomenon5 - direction
	image1 - mode
	phenomenon21 - direction
	star22 - direction
	phenomenon24 - direction
	instrument10 - instrument
	instrument11 - instrument
	star4 - direction
	planet6 - direction
	star1 - direction
	star0 - direction
	star3 - direction
	star9 - direction
	star8 - direction
	instrument12 - instrument
)
(:init
	(supports instrument0 thermograph4)
	(supports instrument0 thermograph0)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 star4)
	(supports instrument1 thermograph3)
	(calibration_target instrument1 star0)
	(supports instrument2 image1)
	(calibration_target instrument2 star4)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(power_avail satellite0)
	(pointing satellite0 star8)
	(supports instrument3 thermograph3)
	(calibration_target instrument3 star1)
	(supports instrument4 image1)
	(calibration_target instrument4 star1)
	(supports instrument5 thermograph3)
	(calibration_target instrument5 star3)
	(supports instrument6 thermograph2)
	(supports instrument6 thermograph0)
	(supports instrument6 image1)
	(calibration_target instrument6 star0)
	(on_board instrument3 satellite1)
	(on_board instrument4 satellite1)
	(on_board instrument5 satellite1)
	(on_board instrument6 satellite1)
	(power_avail satellite1)
	(pointing satellite1 phenomenon21)
	(supports instrument7 thermograph0)
	(calibration_target instrument7 star3)
	(supports instrument8 thermograph4)
	(supports instrument8 thermograph3)
	(supports instrument8 thermograph2)
	(calibration_target instrument8 star3)
	(supports instrument9 thermograph2)
	(supports instrument9 thermograph3)
	(calibration_target instrument9 star1)
	(on_board instrument7 satellite2)
	(on_board instrument8 satellite2)
	(on_board instrument9 satellite2)
	(power_avail satellite2)
	(pointing satellite2 star4)
	(supports instrument10 thermograph2)
	(calibration_target instrument10 star3)
	(supports instrument11 thermograph2)
	(supports instrument11 thermograph4)
	(supports instrument11 thermograph0)
	(calibration_target instrument11 star1)
	(on_board instrument10 satellite3)
	(on_board instrument11 satellite3)
	(power_avail satellite3)
	(pointing satellite3 phenomenon16)
	(supports instrument12 thermograph4)
	(calibration_target instrument12 star3)
	(on_board instrument12 satellite4)
	(power_avail satellite4)
	(pointing satellite4 phenomenon18)
)
(:goal
	(and
		(have_image phenomenon5 thermograph4)
		(have_image planet7 image1)
		(have_image star8 thermograph3)
		(have_image star9 image1)
		(have_image star10 image1)
		(have_image phenomenon13 thermograph2)
		(have_image star15 thermograph2)
		(have_image phenomenon17 thermograph4)
		(have_image phenomenon18 image1)
		(have_image planet19 thermograph2)
		(have_image planet20 thermograph4)
		(have_image phenomenon21 image1)
		(have_image star22 thermograph3)
	)
)
)