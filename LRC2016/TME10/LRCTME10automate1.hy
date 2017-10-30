-- ************************************* LRC - TME10 - 2015-1016 ************************
-- Automate de l'exercice 1 du TD10

-- ***** Définition de l'automate

-- ** Déclaration des variables 
var 
  x: clock;
  y: clock;

-- ** Description de l'automate

automaton exo1
   synclabs: a,b,c,d,fin;
   initially q0 & x=0 & y=0;

-- ses états 

loc q0: while x<3 & y<=1 wait {}
   when x>2 & x<3 sync b goto q1;
   when y>0 & y<= 1 do {y' = 0} sync a goto q0;

loc q1: while True wait {}
    when x > 3 & x<4 & y>1 do {} sync c goto q2;
    when x > 2 & x<3 & x-y<1 do {} sync d goto q3;

loc q2: while True wait {}
    when True sync fin goto q0;
-- hytech ne veut pas d'état aveugle, ajout d'une transition negligeable

loc q3: while True wait {}
     when True  sync fin goto q0;
end


-- ***** Analyse de l'automate

-- ** Déclaration des variables : de type region
var init, accessible, but, but1: region;


-- ** Définition des régions considérées
init:= loc[exo1]=q0 & x=0 & y=0;
but:=  loc[exo1]=q2;
accessible:= reach forward from init endreach;


-- ** Analyse proprement dite 

-- affichage 

prints "on part de " ;
print init ;
prints "on va à ";
print but;              

prints "";                 -- saut de ligne
prints "RESULTAT : ";
print trace to but using accessible; 