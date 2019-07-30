BITMEX_TEST_KEY_ID=${BITMEX_TEST_KEY_ID}
BITMEX_TEST_KEY_SECRET=${BITMEX_TEST_KEY_SECRET}
eval $(docker-machine env crypto-bot)
CONFIG_VERSION=${CONFIG_VERSION} docker stack deploy -c docker-compose.yml crypto-bot --prune
