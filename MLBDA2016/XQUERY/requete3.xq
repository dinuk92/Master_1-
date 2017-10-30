for $i in //open_auctions/auction[position() < 4 ]
return <result id="{$i/@id}"><first>{$i/bidder[1]/increase/text()}</first> <last>{$i/bidder[last()]/increase/text()}</last></result>

