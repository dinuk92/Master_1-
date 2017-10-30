pere(pepin , charlemagne).
mere(berthe, charlemagne).
pere(homer , bart).
mere(marge, bart).
pere(homer , lisa).
mere(marge, lisa).
pere(homer , maggie).
mere(marge, maggie).
pere(dad,homer).

parent(X,Y) :- mere(X,Y).
parent(X,Y) :- pere(X,Y).


parents(X,Y,Z) :- pere(X,Z),mere(Y,Z).

grandpere(X,Y) :- parent(X,Z),pere(Z,Y).

frereOuSoeur(X,Y) :- parent(Z,X),parent(Z,Y),X\=Y.

ancetre(X,Y) :- parent(X,Y).
ancetre(X,Y) :- parent(X,Z),ancetre(Z,Y).






