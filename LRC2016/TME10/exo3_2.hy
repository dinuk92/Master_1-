var 
  x: clock;
  c:analog;

automaton exo32
   synclabs: a5,a10,a20,a50,b,i,d,e,f,fin;
   initially q0 & x=0 ;

loc q0: while x>=0 wait {}
   when True sync  goto p;
   when True sync a10 goto p;
   when True sync a20 goto p;
   when True sync a50 goto p;
   when x = 10 do { x' = 0 } sync d goto q0;
   when c >= 40 & x<= 10 do { x' = 0 } sync b goto q1;	 

loc p5 : while x >= 1 wait {c=5}
	when x=1 do { x' = 0} sync e goto q0;
loc p10 : while x >= 1 wait {c=10}
	when x=1 do { x' = 0} sync e goto q0;
loc p20 : while x >= 1 wait {c=20}
	when x=1 do { x' = 0} sync e goto q0;
loc p50 : while x >= 1 wait {c=50}
	when x=1 do { x' = 0} sync e goto q0;
	
loc q1: while True wait {}
   when x=1 do { x' = 0 } sync i goto q2;

loc q2: while True wait {}
   when x = 30 do { x' = 0 } sync f goto q0;
end

var init, accessible, but, but1: region;

init:= loc[exo32]=q0 & x=0 ;
but:=  loc[exo32]=q2 & x=30 ;
accessible:= reach forward from init endreach; 

prints "on part de " ;
print init ;
prints "on va à ";
print but;              

prints "";                 -- saut de ligne
prints "RESULTAT : ";
print trace to but using accessible;
