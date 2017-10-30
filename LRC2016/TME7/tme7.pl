subs(A,B):-equiv(A,B).
subs(A,B):-equiv(B,A).

subs(chat,felin).
subs(lion,felin).
equiv(chien,canide).
subs(souris,mammifere).
subs(felin,mammifere).
subs(canide,mammifere).
subs(mammifere, animal).
subs(canari,animal).
subs(animal,etreVivant).
subs(and(animal,plante),nothing).
subs(and(animal,some(aMaitre)),pet).
subs(pet,some(aMaitre)).
subs(some(aMaitre),all(aMaitre,humain)). 
subs(chiuhahua,and(chien, pet)).
equiv(carnivoreExc,all(mange,animal)).
equiv(herbivoreExc,all(mange,plante)).
subs(lion,carnivoreExc).
subs(carnivoreExc,predateur).
subs(animal,some(mange)).
subs(and(some(mange),all(mange , nothing)),nothing).
inst(felix,chat).
inst(pierre,humain).
instR(felix,aMaitre,pierre).
instR(princesse, chiuhahua).
inst(marie,humain).
instR(princesse,aMaitre,marie).


 


subsS1(C,C).
subsS1(C,D):-subs(C,D),C\==D.
subsS1(C,D):-subs(C,E),subsS1(E,D).

subsS(C,D):- subsS(C,D,[C]).
subsS(C,C,_).
subsS(C,D,_):-subs(C,D),C\==D.
subsS(C,D,L):-subs(C,E),not(member(E,L)),subsS(E,D,[E|L]),E\==D.


