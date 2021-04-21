:- op(800, fx, if).
:- op(800, xfy, <=).
:- op(700, xfx, then).
:- op(300, xfy, and).
:- op(200, xfy, or).

:- dynamic(fact/1).

if
 hall_wet and kitchen_dry
then
 leak_in_bathroom.

if
 hall_wet and bathroom_dry
then
 problem_in_kitchen.

if
 window_closed or no_rain
then
 no_water_from_outside.

if
 problem_in_kitchen and no_water_from_outside
then
 leak_in_kitchen.


fact(hall_wet).
fact(bathroom_dry).
fact(window_closed).

% :- is_true(leak_in_kitchen).



















