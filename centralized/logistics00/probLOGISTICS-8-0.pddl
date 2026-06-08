(define (problem logistics-8-0) (:domain logistics)
(:objects
	tru1 - truck
	tru3 - truck
	tru2 - truck
	pos2 - location
	pos3 - location
	pos1 - location
	obj21 - package
	obj22 - package
	obj23 - package
	apn1 - airplane
	apt3 - airport
	apt2 - airport
	apt1 - airport
	obj33 - package
	obj32 - package
	obj31 - package
	cit1 - city
	cit2 - city
	cit3 - city
	obj11 - package
	obj13 - package
	obj12 - package
)
(:init
	(at apn1 apt1)
	(at tru1 pos1)
	(at obj11 pos1)
	(at obj12 pos1)
	(at obj13 pos1)
	(at tru2 pos2)
	(at obj21 pos2)
	(at obj22 pos2)
	(at obj23 pos2)
	(at tru3 pos3)
	(at obj31 pos3)
	(at obj32 pos3)
	(at obj33 pos3)
	(in-city tru1 pos1 cit1)
	(in-city tru1 apt1 cit1)
	(in-city tru2 pos2 cit2)
	(in-city tru2 apt2 cit2)
	(in-city tru3 pos3 cit3)
	(in-city tru3 apt3 cit3)
)
(:goal
	(and
		(at obj11 pos3)
		(at obj21 pos2)
		(at obj31 apt3)
		(at obj22 pos3)
		(at obj12 pos1)
		(at obj23 apt2)
		(at obj13 apt2)
		(at obj32 apt1)
	)
)
)