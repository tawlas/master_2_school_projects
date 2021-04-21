female(helen).
female(ruth).
female(petunia).
female(lili).

male(paul).
male(albert).
male(vernon).
male(james).
male(dudley).
male(harry).

parent_of(paul,petunia).
parent_of(helen,petunia).
parent_of(paul,lili).
parent_of(helen,lili).
parent_of(albert,james).
parent_of(ruth,james).
parent_of(petunia,dudley).
parent_of(vernon,dudley).
parent_of(lili,harry).
parent_of(james,harry).

% Définition 1
ancestor_of(X,Y) :- parent_of(X,Y).
ancestor_of(X,Y) :- parent_of(X,Z),
                    ancestor_of(Z,Y).

% Définition 2
% ancestor_of(X,Y) :- parent_of(X,Z),
%                     ancestor_of(Z,Y).
% ancestor_of(X,Y) :- parent_of(X,Y).

% Définition 3
% ancestor_of(X,Y) :- parent_of(X,Y).
% ancestor_of(X,Y) :- ancestor_of(Z,Y),
%                     parent_of(X,Z).

% Définition 4
% ancestor_of(X,Y) :- ancestor_of(Z,Y),
%                     parent_of(X,Z).
% ancestor_of(X,Y) :- parent_of(X,Y).