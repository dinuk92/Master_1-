<bib>
{for $b in //book where count($b/author) > 0
return <book>{$b/title}
             {for $a in $b/author[position()<=2]
              return $a}
             {if (count($b/auteur)>2) then <et_al/> else()} 
       </book>
}
</bib>