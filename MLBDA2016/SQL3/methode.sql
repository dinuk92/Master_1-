-- compléter l'entête 
-- ==================

-- NOM    :
-- Prénom :

-- NOM    :
-- Prénom :

-- Groupe :
-- binome :

-- ================================================
set sqlbl on
create type body piece_de_base as
member function masse return number is
res number
begin
select deref(mat).masse_volumique into res from dual

return self.volume() * res 
end;
end;


create type body spherique as
member function volume return number is
       res number
begin
	res = (4/3)*self.rayon*self.rayon*self.rayon*3.14
	return res ;   
end ;
end ;
create type body cubique as
member function volume return number is
       res number
begin
	res = cote * cote *cote ;
	return res  ;  
end ;
end ;
create type body cylindrique as
member function volume return number is
       res number
begin
	res =  3.14 * rayon * rayon  * hauteur ;
	return res;
end ;
end ;

create type body piece_composite as
member function volume return number is
       res number
begin
	select  sum (e.qte*value(ensp).volume()) into res
	from table(enspiece) e , table( value(e).tpi) ensp ; 
	return res;
end ;
member function masse return number is
       res number
begin
	select  sum (e.qte*value(ensp).masse()) into res
	from table(enspiece) e , table( value(e).tpi) ensp ; 
	return res;
end ;

