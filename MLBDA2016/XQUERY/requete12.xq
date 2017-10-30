<bib>{
  for $i in doc("bib.xml")//book, 
      $j in doc("bib.xml")//book[position() > $i/position()]
      return 
      if($i/title!=$j/title and deep-equal($i/author,$j/author))
      then <book-pair> {$i/title} {$j/title} </book-pair>
      else() 
}</bib>