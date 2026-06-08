(define (domain wireless)
	(:requirements :strips :typing)
(:types
	base sensor - node
	node level message - object
)
(:constants
	Zero Low Normal High - level
)
(:predicates
	(neighbor ?n1 - node ?n2 - node)
	(has-data ?n - node ?s - sensor)
	(higher ?l1 - level ?l2 - level)
	(next ?l1 - level ?l2 - level)
	(is-message-at ?m - message ?n - node)
	(not-message-at ?m - message ?n - node)
	(message-data ?m - message ?s - sensor)
	(not-message-data ?m - message ?s - sensor)
	(sending ?from - sensor ?to - node ?m - message)
	(not-sending ?from - sensor ?to - node ?m - message)
	(energy ?s - sensor ?lv - level)
)

(:action generate-data
	:parameters (?s - sensor ?e0 - level ?e1 - level)
	:precondition (and
		(energy ?s ?e0)
		(higher ?e0 Zero)
		(next ?e0 ?e1)
	)
	:effect (and
		(not (energy ?s ?e0))
		(energy ?s ?e1)
		(has-data ?s ?s)
	)
)


(:action add-to-message
	:parameters (?s1 - sensor ?s2 - sensor ?m - message)
	:precondition (and
		(has-data ?s1 ?s2)
		(is-message-at ?m ?s1)
		(not-message-data ?m ?s2)
	)
	:effect (and
		(not (has-data ?s1 ?s2))
		(message-data ?m ?s2)
		(not (not-message-data ?m ?s2))
	)
)


(:action send-message
	:parameters (?sender - sensor ?receiver - node ?m - message ?e0 - level ?e1 - level)
	:precondition (and
		(energy ?sender ?e0)
		(higher ?e0 Zero)
		(next ?e0 ?e1)
		(neighbor ?sender ?receiver)
		(is-message-at ?m ?sender)
		(not-message-at ?m ?receiver)
		(not-sending ?sender ?receiver ?m)
	)
	:effect (and
		(not (energy ?sender ?e0))
		(not (is-message-at ?m ?sender))
		(not-message-at ?m ?sender)
		(energy ?sender ?e1)
		(sending ?sender ?receiver ?m)
		(not (not-sending ?sender ?receiver ?m))
	)
)


(:action receive-message
	:parameters (?receiver - node ?sender - sensor ?m - message)
	:precondition (and
		(not-message-at ?m ?receiver)
		(sending ?sender ?receiver ?m)
	)
	:effect (and
		(not (sending ?sender ?receiver ?m))
		(not-sending ?sender ?receiver ?m)
		(is-message-at ?m ?receiver)
		(not (not-message-at ?m ?receiver))
	)
)


(:action get-data-from-message
	:parameters (?n - node ?s - sensor ?m - message)
	:precondition (and
		(is-message-at ?m ?n)
		(message-data ?m ?s)
	)
	:effect (and
		(not (message-data ?m ?s))
		(not-message-data ?m ?s)
		(has-data ?n ?s)
	)
)

)