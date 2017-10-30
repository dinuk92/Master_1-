-- compléter l'entête 
-- ==================

-- NOM    :Wijesinghe
-- Prénom :Dinuk

-- NOM    :
-- Prénom :

-- Groupe :
-- binome :

-- ================================================

-- stockage des données : définition des relations
-- ====================



-- instanciation des objets
-- ========================

desc les_piece_base;

desc lesmatiere;

insert into lesmatiere values(matiere('bois',10,2));


insert into lesmatiere values(matiere('fer',5,3));

insert into lesmatiere values(matiere('ferrite',6,10));

select * from lesmatiere;

drop table lesmatiere;


insert into les_piece_base values( cylindrique('canne',(select ref(m)
       	    		   	   			from lesmatiere m
							where m.nom='bois'),
							30,
							2));

insert into les_piece_base values( cylindrique('clou',(select ref(m)
       	    		   	   			from lesmatiere m
							where m.nom='fer'),
							20,
							1));

insert into les_piece_base values( cylindrique('aimant',(select ref(m)
       	    		   	   			from lesmatiere m
							where m.nom='ferrite'),
							5,
							2));

insert into les_piece_base values(paralele('plateau',(select ref(m)
       	    		   	   			from lesmatiere m
							where m.nom='bois'),1,100,80));

insert into les_piece_base values(spherique('boule',(select ref(m)
       	    		   	   			from lesmatiere m
							where m.nom='fer'),30));

insert into les_piece_base values(spherique('pied',(select ref(m)
       	    		   	   			from lesmatiere m
							where m.nom='bois'),30));



--dumb tests
--select p.nom ,p.mat.*  from les_piece_base p;

select p.nom ,p.mat.nom, p.mat.prix_kilo,p.mat.masse_volumique  from les_piece_base p;

select p.nom ,deref(p.mat)  from les_piece_base p;




insert into les_piece_composite values
(piece_composites(100,enspiece(tenspiece((select ref(p) from les_piece_base p where p.nom='plateau'),1),(tenspiece((select ref(p) from les_piece_base p where p.nom='pied'),4)),(tenspiece((select ref(p) from les_piece_base p where p.nom='clou'),12))))));

insert into les_piece_composite values
(piece_composites(10,enspiece(
tenspiece((select ref(p) from les_piece_base p where p.nom='table'),1),
tenspiece((select ref(p) from les_piece_base p where p.nom='boule'),3),
tenspiece((select ref(p) from les_piece_base p where p.nom='canne'),2))));

			  
	        
