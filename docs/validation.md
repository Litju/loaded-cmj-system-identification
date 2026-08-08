# Validation

Validation compares forward predictions with public observations in physical
units. It includes force-platform RMSE and maximum error, bar displacement and
velocity errors, scalar mechanics summaries, movement/takeoff/landing timing,
finite-state checks, contact duration, phase ordering, and countermovement
depth.

The source event interpretation is retained: movement onset follows the
weighing rule, takeoff requires a sustained bilateral no-contact interval after
propulsion, and landing requires sustained recontact. A missing physical
takeoff is represented as `None`; it is never replaced by a scheduled phase
time.

