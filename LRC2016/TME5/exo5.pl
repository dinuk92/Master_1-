et(0,0,0).
et(0,1,0).
et(1,0,0).
et(0,1,1).
et(1,1,1).
ou(0,0,0).
ou(0,1,0).
ou(1,0,0).
ou(1,1,1).
non(1,0).
non(0,1).


nand(X,Y,Z):-et(X,Y,W),non(W,Z).

xor(x,y,z) :-ou(X,Y,Z),et(X,Y,V),non(V,W),et(V,W,Z).

circuit(X,Y,Z):-nand(X,Y,Z),non(X,X2),xor(Z,X2,X3),non(X3,Z).

