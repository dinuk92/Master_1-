concatenate([],L,L).
concatenate([X|L],L2,[X|L3]) :- concatenate(L,L2, L3).

inverse([],[]).
inverse([X|L1],L2):-inverse(L1,L3),concatenate(L3,[X],L2).

supprime([],Y,[]).
supprime([T|L],Y,[T|W]):- supprime(L,Y,W), T \== Y.
supprime([Y|L],Y,Z):- supprime(L,Y,Z).

filtre(L,[],L).
filtre(L1,[X|Y],L2):- supprime(L1,X,L3),filtre(L3,Y,L2).

palindrome([]).
palindrome(L):- inverse(L,L). 

palindrome2([X|T]):- concatenate(P,[X],T),palindrome2(P).
palindrome2([]).
palindrome2([_]).
