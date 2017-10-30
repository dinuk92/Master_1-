<bib>
{for $i in doc("bib.xml")//book
  let $a:=$i/author
  return     
  if(exists($a))then
    <book>{$i/title}{$a}</book>
  else
    <reference>{$i/title}{$i/editor/affiliation}</reference>     
}
</bib>