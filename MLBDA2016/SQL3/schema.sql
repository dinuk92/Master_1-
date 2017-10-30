-- compléter l'entête 
-- ==================

-- NOM    : Wijesinghe 
-- Prénom : Dinuk

-- NOM    :
-- Prénom :

-- Groupe :
-- binome :

-- ================================================

set sqlbl on

-- nettoyer le compte
@vider

-- Définition des types de données

prompt essai

create type T_un_type as object (
 un_attribut Number
);
@compile

create type piece as object (
       nom varchar2(20)
       
) not final;
/


create or replace type matiere as object (
       nom varchar2(20),
       prix_kilo number,
       masse_volumique number
)not final;
/

create or replace  type piece_base under piece (
       mat ref  matiere,
       not instantiable member function volume return number,
       member function masse return number
)NOT INSTANTIABLE not final;
@compile

create type cubique under piece_base (
       cote number,
       overriding member function volume return number
);
@compile

create type spherique under piece_base (
       rayon number,
       overriding member function volume return number
);
/

create type cylindrique under piece_base (
       hauteur number,
       rayon number,
       overriding member function volume return number 
);
/

create type paralele under piece_base(
       longueur number,
       largeur number,
       hauteur number,
       overriding member function volume return number
);
/

create type tenspiece as object (
      tpi ref piece,
      qte number
);
/

create type EnsPiece as table of tenspiece;
@compile

create type piece_composites under piece (
       cout_ass number,
       ens_piece EnsPiece,
       member function volume return number,
       member function masse return number
);
@compile

create table lesmatiere of matiere ;

create table les_piece_base of piece_base ;

create table les_piece_composite of piece_composites 
nested table ens_piece store as t_table;

desc piece_composites
desc enspiece

--insert into table (



-- liste de tous les types créés
@liste

