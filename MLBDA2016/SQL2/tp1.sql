-- MABD TP1 SQL avec la base MONDIAL


-- -------------------- binome -------------------------
-- NOM Wijesinghe
-- Prenom Dinuk	

-- NOM
-- Prenom
-- -----------------------------------------------------

-- pour se connecter à oracle:
-- sqlplus E1234567/E1234567@oracle
-- remplacer E12345657 par la lettre E suivie de votre numéro de login

set sqlbl on
set linesize 150

prompt schema de la table Country
desc Country

prompt schema de la table City
desc City

prompt schema de la table IsMember
desc IsMember

prompt schema de la table City
desc City

desc  borders

-- pour afficher un nuplet entier sur une seule ligne
column name format A15
column capital format A15
column province format A20

-- Requete 0

select * from Country where name = 'France';

------------

--1)
select p.name
from Country p , Organization o, isMember i
where p.code  = i.country and i.organization = o.abbreviation and o.abbreviation = 'UN'
group by p.Name
order by p.Name;

--2)

select p.name
from Country p , Organization o, isMember i
where p.code  = i.country and i.organization = o.abbreviation and o.abbreviation = 'UN'
group by p.name, p.population
order by p.population desc ;

--Pourquoi ca marche pas de maniere imbriquée? 
select bdd.name
from ( select p.name
from Country p , Organization o, isMember i
where p.code  = i.country and i.organization = o.abbreviation and o.abbreviation = 'UN'
group by p.name, p.population
order by p.population desc 
) as bdd ;

--3

select p.name
from Country p , Organization o, isMember i
where p.code  = i.country and i.organization = o.abbreviation and o.abbreviation != 'UN'
group by p.name, p.population
order by p.population desc ;

--4

select p.name
from Country p , borders b 
where b.country1 = 'F'
and b.country2 = p.code 
union
select p2.name
from Country p2, borders b2  
where b2.country1 = p2.code
and b2.country2 = 'F'
;

--5

select p.name
from Country p , borders b 
where( b.country1 = 'F'and b.country2 = p.code) or (b.country1 = p.code and b.country2 = 'F') 
;

--6

select sum(b.length)
from Country p , borders b 
where( b.country1 = 'F'and b.country2 = p.code) or (b.country1 = p.code and b.country2 = 'F')
;

--7

select p.nom, count(
       select b.country2
       from country p2, borders b
       where b.country1 = p2.code and p=p2)
from country p
group by p.nom
;


select p.nom, count(
       select *
       from borders b
       where b.country1 = p.code)
from country p 
group by p.nom
;


select count(b.country2)
from country p, borders b
where b.country1=p.code
group by p.name
;

-- 8

select p.name, sum(p.population)
from country p, borders b
where b.country1=p.code
group by p.name
;

--9


select p.name, sum(p.population) as population
from country p, borders b, organization o, isMember i
where b.country1=p.code and o.abbreviation = i.organization and i.country=p.code and o.abbreviation='EU' 
group by p.name
;

--10

select o.abbreviation , count(p.code),sum(p.population)  
from organization o , isMember i , country p 
where o.abbreviation = i.organization and i.country = p.code 
group by o.abbreviation
;

--11

select o.abbreviation , count(p.code),sum(p.population)  
from organization o , isMember i , country p 
where o.abbreviation = i.organization and i.country = p.code  
group by o.abbreviation
having count(p.code) >100
;


--12 ? ? ?

select c.name ,
from Continent c
;

select c.mountains
from mountain c
;


select distinct( e.country) , m.name , max(m.height) 
from Mountain m , geo_mountain gm , encompasses e , Continent c     
where c.name='America' and c.name = e.continent and gm.country = e.country
group by e.country, m.name , m.height
;

--13

select r.name
from river r 
where r.river = 'Nile'
;

--14

select r.name
from river r 
where r.river = 'Nile'
union
select r3.name
from river r2 , river r3
where r2.river ='Nile'
and r3.river=r2.name
union
select r4.name
from river r4 , river r5 , river r6
where r6.river ='Nile'
and r5.river=r6.name
and r4.river=r5.name
;

--15


select sum(r2.Length)
from River r1, River r2
where r1.name='Nile'and r2.river=r1.name or r2.name='Nile';  

--16a)
-->= all ou mettre count(max(... 
select o.name
from Organization o, IsMember i, Country c 
where i.country=c.code and i.organization=o.abbreviation
group by o.name
having count(c.name)>= all (select count(c2.name)
       		       from country c2, organization o2, ismember i
		       where c2.code = i.country and i.organization=o2.abbreviation
		       group by o2.abbreviation
		       )
;
--16b)
select *
from (select o.abbreviation, count(p.code)
from organization o , isMember i , country p 
where o.abbreviation = i.organization and i.country = p.code 
group by o.abbreviation
order by count(p.code) desc)
where rownum <=3;

--17)

select c.name 
from country c, borders b1
where c.name='Algeria'
or c.name='Libya'
or c.code=b1.country1 and (b1.country2='DZ'or b1.country2='LAR')
or c.code=b1.country2 and (b1.country1='DZ'or b1.country1='LAR'); 



/****/
select sum(c.population)/sum(c.area)
from
(select b1.country1
from Borders b1
where ((b1.country1='DZ')or (b1.country2='DZ')) or ((b1.country1='LAR')or (b1.country2='LAR'))
union
select b1.country2
from Borders b1
where ((b1.country1='DZ')or (b1.country2='DZ')) or ((b1.country1='LAR')or (b1.country2='LAR'))
) t, country c
where c.code=t.country1;

select sum(c.population)/sum(c.area)
from
(select b1.country1
from Borders b1
where b1.country2='DZ' or b1.country2='LAR'
union
select b1.country2
from Borders b1
where b1.country1='DZ' or b1.country1='LAR'
) t, country c
where c.code=t.country1;

select * from MOndial.r17;

--18)
select sum(c.population)/(sum(c.area)-sum(d.area))
from
(select b1.country1
from Borders b1
where ((b1.country1='DZ')or (b1.country2='DZ')) or ((b1.country1='LAR')or (b1.country2='LAR'))
union
select b1.country2
from Borders b1
where ((b1.country1='DZ')or (b1.country2='DZ')) or ((b1.country1='LAR')or (b1.country2='LAR'))
) t, country c , geo_desert gd , desert d 
where c.code=t.country1
and gd.country=c.code
and gd.desert=d.name
;

select * from mondial.r18;


--19)
select sum(c.population)
from country c
;

select t.nom, sum(c2.population), sum(t.p)/sum(c2.population)
from (select r.name as nom, c.name, r.percentage*c.population as p
     from religion r, country c
     where r.country=c.code) t, country c2
group by t.nom
;


select r.name as nom, sum(r.percentage*c.population)/(select sum(c.population) from country c)/100 as pop
from religion r, country c
where r.country=c.code
group by r.name;

select *  from mondial.r19;

--20

select c.name , c2.name
from country c , country c2  , Encompasses e , geo_sea gs 
where e.continent = 'Europe'
and e.country = c2.code
and e.country = c.code 
and c.code = gs.country 
and not exists (select *
    	     	from geo_sea gs2 , encompa
		where
		and e2.country = c2.code
		and gs2.sea != gs.sea
		)
;


select c.name , gs.sea
from encompasses e , country c , geo_sea gs
where e.country = c.code 
and gs.country = c.code
and e.continent = 'Europe';


select distinct s.name
from sea s, geo_sea g, encompasses e, country c
where e.country=c.code and e.continent='Africa' and g.country=c.code and g.sea=s.name
and s.depth = 
(select  MAX(s1.depth)
from sea s1, geo_sea g1, encompasses e1, country c1
where e1.country=c1.code and e1.continent='Africa' and g1.country=c1.code and g1.sea=s1.name);


select distinct s.name
from sea s, geo_sea g, encompasses e, country c
where e.country=c.code and e.continent='Africa' and g.country=c.code and g.sea=s.name
and s.depth >= ALL 
(select  s1.depth
from sea s1, geo_sea g1, encompasses e1, country c1
where e1.country=c1.code and e1.continent='Africa' and g1.country=c1.code and g1.sea=s1.name);


select c2.name, b.length as longueur_frontiere
from country c1, country c2, borders b
where c1.name = 'France'
and ((c1.code = b.country1 and c2.code = b.country2)
or
(c1.code = b.country2 and c2.code = b.country1))
and b.length >= ALL (select  b1.length
from  country c3, borders b1
where  ((c1.code = b1.country1 and c3.code = b1.country2)
or
(c1.code = b1.country2 and c3.code = b1.country1)));


select c2.name, b.length as longueur_frontiere
from country c1, country c2, borders b
where c1.name = 'France'
and ((c1.code = b.country1 and c2.code = b.country2)
or
(c1.code = b.country2 and c2.code = b.country1))
and b.length = (select MAX( b1.length)

select c1.name, c2.name
from country c1, country c2, borders b, language l1, language l2
where b.country1=c1.code and b.country2=c2.code and l1.country=c1.code and l2.country=c2.code and 
l1.percentage > 30 and l2.percentage > 30 and l1.name=l2.name ;
