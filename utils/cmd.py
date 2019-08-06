import argparse


def default_parser():
    parser = argparse.ArgumentParser(description='Market data downloader')
    parser.add_argument('-s', '--symbols',
                        type=str,
                        nargs='+',
                        required=True,
                        help='The symbol of the instrument/currency pair')

    parser.add_argument('-e','--exchange',
                        type=str,
                        help='The exchange to download from')

    parser.add_argument('-t','--timeframe',
                        type=str,
                        default='1m',
                        choices=['10s', '30s', '1m', '5m','15m', '30m','1h', '2h', '3h', '4h', '6h', '12h', '1d', '1M', '1y'],
                        help='The timeframe to download')

    parser.add_argument('-days', '--days',
                         type=int,
                         help='The number of days to fetch ohlcv'
                        )

    parser.add_argument('-from', '--from_date',
                         type=str,
                         help='The date from which to start dowloading ohlcv from'
                        )

    parser.add_argument('-to', '--to_date',
                         type=str,
                         help='The date up to which to download ohlcv to'
                        )

    parser.add_argument('--debug',
                            action ='store_true',
                            help=('Print Sizer Debugs'))

    return parser


def parse_args():
    parser = argparse.ArgumentParser(description='Market data downloader')
    parser.add_argument('-s', '--symbols',
                        type=str,
                        nargs='+',
                        required=True,
                        help='The symbol of the instrument/currency pair')

    parser.add_argument('-e','--exchange',
                        type=str,
                        help='The exchange to download from')

    parser.add_argument('-t','--timeframe',
                        type=str,
                        default='1m',
                        choices=['10s', '30s', '1m', '5m','15m', '30m','1h', '2h', '3h', '4h', '6h', '12h', '1d', '1M', '1y'],
                        help='The timeframe to download')

    parser.add_argument('-days', '--days',
                         type=int,
                         help='The number of days to fetch ohlcv'
                        )

    parser.add_argument('-from', '--from_date',
                         type=str,
                         help='The date from which to start dowloading ohlcv from'
                        )

    parser.add_argument('-to', '--to_date',
                         type=str,
                         help='The date up to which to download ohlcv to'
                        )

    parser.add_argument('--debug',
                            action ='store_true',
                            help=('Print Sizer Debugs'))

    return parser.parse_args()






def parse_arima_args():
    parser = argparse.ArgumentParser(description='Market data downloader')


    parser.add_argument('-s','--symbols',
                        type=str,
                        nargs='+',
                        required=True,
                        help='The Symbol of the Instrument/Currency Pair To Download')

    parser.add_argument('-e','--exchange',
                        type=str,
                        help='The exchange to download from')

    parser.add_argument('-p', '--p_order',
                        type=int,
                        default=0,
                        help='The ARIMA model p order')

    parser.add_argument('-q', '--q_order',
                        type=int,
                        default=0,
                        help='The ARIMA model q order')

    parser.add_argument('d', '--d_order',
                        type=int,
                        default=0,
                        help='The ARIMA model d order')

    parser.add_argument('-t','--timeframe',
                        type=str,
                        default='5m',
                        choices=['1m', '5m','15m', '30m','1h', '2h', '3h', '4h', '6h', '12h', '1d', '1M', '1y'],
                        help='The timeframe to download')

    parser.add_argument('-days', '--days',
                         type=int,
                         help='The number of days to fetch ohlcv'
                        )

    parser.add_argument('-from', '--from_date',
                         type=str,
                         help='The date from which to start dowloading ohlcv from'
                        )

    parser.add_argument('-end', '--to_date',
                         type=str,
                         help='The date up to which to download ohlcv to'
                        )

    parser.add_argument('--debug',
                            action ='store_true',
                            help=('Print Sizer Debugs'))

    return parser.parse_args()

