for $i in //book[@year gt "1991" ][ publisher="Addison-Wesley"] 
  return <book year="{$i/@year}">
  {
   let $cpt:=$i/title
   return $cpt
  }
  </book>