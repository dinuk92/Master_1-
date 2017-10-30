<results>{
for $i in //book/title
for $j in //book[title=$i]/author

return <result>{$i}{$j}</result>
}
</results>