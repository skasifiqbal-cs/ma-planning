(define (problem DLOG-3-2-5) (:domain driverlog)
(:objects
	p1-2 - location
	driver2 - driver
	truck1 - truck
	truck2 - truck
	driver1 - driver
	s2 - location
	s1 - location
	s0 - location
	p0-2 - location
	p0-1 - location
	driver3 - driver
	package1 - package
	package2 - package
	package3 - package
	package4 - package
	package5 - package
)
(:init
	(at driver1 s1)
	(at driver2 s0)
	(at driver3 s0)
	(at truck1 s1)
	(empty truck1)
	(at truck2 s1)
	(empty truck2)
	(at package1 s0)
	(at package2 s0)
	(at package3 s2)
	(at package4 s2)
	(at package5 s1)
	(path s0 p0-1)
	(path p0-1 s0)
	(path s1 p0-1)
	(path p0-1 s1)
	(path s0 p0-2)
	(path p0-2 s0)
	(path s2 p0-2)
	(path p0-2 s2)
	(path s1 p1-2)
	(path p1-2 s1)
	(path s2 p1-2)
	(path p1-2 s2)
	(link s0 s1)
	(link s1 s0)
	(link s0 s2)
	(link s2 s0)
	(link s1 s2)
	(link s2 s1)
)
(:goal
	(and
		(at truck1 s2)
		(at truck2 s2)
		(at package1 s1)
		(at package2 s1)
		(at package3 s1)
		(at package4 s0)
		(at package5 s1)
	)
)
)