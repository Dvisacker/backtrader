# Create docker machines
docker-machine create --driver amazonec2 \
 --amazonec2-open-port 7946/tcp \
 --amazonec2-open-port 7946/udp \
 --amazonec2-open-port 4789/tcp \
 --amazonec2-open-port 4789/udp \
 --amazonec2-instance-type="t2.micro" \
  --amazonec2-region ap-northeast-2 \
  crypto-bot

eval $(docker-machine env crypto-bot)
docker swarm init --advertise-addr $(docker-machine ip crypto-bot)

eval $(docker-machine env -u)
ip=$(docker-machine ip crypto-bot)
docker network create --driver overlay --attachable crypto-bot
