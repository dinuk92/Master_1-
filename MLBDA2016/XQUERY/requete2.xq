for $i in //open_auctions/auction[position() lt 4 ]
return <result id="{$i/@id}">{$i/initial}</result>