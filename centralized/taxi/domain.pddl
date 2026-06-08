(define (domain taxi)
	(:requirements :strips :typing)
(:types
	location agent - object
	taxi passenger - agent
)
(:predicates
	(directly-connected ?l1 - location ?l2 - location)
	(at ?a - agent ?l - location)
	(in ?p - passenger ?t - taxi)
	(empty ?t - taxi)
	(free ?l - location)
	(goal-of ?p - passenger ?l - location)
)

(:action drive
	:parameters (?t - taxi ?from - location ?to - location)
	:precondition (and
		(at ?t ?from)
		(directly-connected ?from ?to)
		(free ?to)
	)
	:effect (and
		(not (at ?t ?from))
		(not (free ?to))
		(at ?t ?to)
		(free ?from)
	)
)


(:action enter
	:parameters (?p - passenger ?t - taxi ?l - location)
	:precondition (and
		(at ?p ?l)
		(at ?t ?l)
		(empty ?t)
	)
	:effect (and
		(not (empty ?t))
		(not (at ?p ?l))
		(in ?p ?t)
	)
)


(:action exit
	:parameters (?p - passenger ?t - taxi ?l - location)
	:precondition (and
		(in ?p ?t)
		(at ?t ?l)
		(goal-of ?p ?l)
	)
	:effect (and
		(not (in ?p ?t))
		(empty ?t)
		(at ?p ?l)
	)
)

)