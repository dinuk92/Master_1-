-- compléter l'entête 
-- ==================

-- NOM    :Wijesinghe Dinuk
-- Prénom :

-- NOM    :
-- Prénom :

-- Groupe :
-- binome :

-- ================================================

-- suppression des types créés
-- ==========================

-- drop type Etudiant;
-- drop type Adresse;
-- drop type Cercle;
-- drop type Polygone;
-- drop type Point;


-- définition des types 
-- ====================

-- un étudiant décrit par les attributs nom, prénom, diplôme,
create type Etudiant as Object
(nom Varchar2(10),
prenon Varchar2(10),
diplome Varchar2(100) 
);


-- un module d'enseignement décrit par les attributs nom, diplôme,
create type Module as Object
(nom Varchar2(10),
diplome Varchar2(100) 
);

-- un point du plan euclidien,
create type point as object
(x number(2),
y number(3)
);

-- un cercle,
create type cercle as object 
( centre point,
 rayon number(2)
);

-- une adresse,

create type addresse as object
( numero number(3),
  rue number(2),
  cp number(5),
  ville varchar(100),
);

-- type candidat
create type Candidat as object
( nom varchar(10),
prenom varchar(10),
age number(2),
add addresse 
);

--type filière
create type Filiere as table of module;

--type polygone
create type polygone as Varray()of point;


create table lesCandidat of Candidat ;

create table Carrelage (
       poly polygone;
       col val;
       num number(100) ;
);

insert into lesCandidat values (  
       Candidat( 'Dupont','Jean','20', adresse('20','rue du Bac',75007, 'Paris')));

insert into carrelage(
 
);
