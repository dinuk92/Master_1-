var 
  x: clock;
  y: analog;

automaton exo31
   synclabs: a,b,c,d;
   initially q0 & x=0 & y=0;

loc q0: while True wait {dy=0}
   when True do {x' = 0} sync a goto q1;
  

loc q1: while x<=10 wait {dy=0}
   when x<10 do {x' = 0 } sync b goto q2;
   when x<=0 sync c goto q0;

loc q2: while x<=30 wait {dy= 1}
   when x<30 sync d goto q0;
end

var init, accessible, but, but1: region;

init:= loc[exo31]=q0 & x=0 ;
but:=  loc[exo31]=q0 & y=1 ;
accessible:= reach forward from init endreach; 

prints "on part de " ;
print init ;
prints "on va à ";
print but;              

prints "";                 -- saut de ligne
prints "RESULTAT : ";
print trace to but using accessible; 
