import numpy as np
import pandas as pd

from zipline import api
from zipline.assets import Equity, Future
from zipline.assets.synthetic import make_commodity_future_info
from zipline.testing import (
    parameter_space,
    prices_generating_returns,
    simulate_minutes_for_day,
)
from zipline.testing.fixtures import (
    WithMakeAlgo,
    WithConstantEquityMinuteBarData,
    WithConstantFutureMinuteBarData,
    WithWerror,
    ZiplineTestCase,
)
from zipline.testing.predicates import assert_equal, wildcard


def T(cs):
    return pd.Timestamp(cs, tz='utc')


def portfolio_snapshot(p):
    """Extract all of the fields from the portfolio as a new dictionary.
    """
    fields = (
        'cash_flow',
        'starting_cash',
        'portfolio_value',
        'pnl',
        'returns',
        'cash',
        'positions',
        'positions_value',
        'positions_exposure',
    )
    aliases = {
        'cash_flow': 'capital_used',
        'cash': 'ending_cash',
        'positions_value': 'ending_value',
        'positions_exposure': 'ending_exposure',
    }

    return {
        aliases.get(field, field): getattr(p, field)
        for field in fields
    }


class TestConstantPrice(WithConstantEquityMinuteBarData,
                        WithConstantFutureMinuteBarData,
                        WithMakeAlgo,
                        WithWerror,
                        ZiplineTestCase):
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = True
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE = True

    ASSET_FINDER_EQUITY_SIDS = [ord('A')]

    EQUITY_MINUTE_CONSTANT_LOW = 1.0
    EQUITY_MINUTE_CONSTANT_OPEN = 1.0
    EQUITY_MINUTE_CONSTANT_CLOSE = 1.0
    EQUITY_MINUTE_CONSTANT_HIGH = 1.0
    EQUITY_MINUTE_CONSTANT_VOLUME = 100.0

    FUTURE_MINUTE_CONSTANT_LOW = 1.0
    FUTURE_MINUTE_CONSTANT_OPEN = 1.0
    FUTURE_MINUTE_CONSTANT_CLOSE = 1.0
    FUTURE_MINUTE_CONSTANT_HIGH = 1.0
    FUTURE_MINUTE_CONSTANT_VOLUME = 100.0

    START_DATE = T('2014-01-06')
    END_DATE = T('2014-01-10')

    # note: class attributes after this do not configure fixtures, they are
    # just used in this test suite

    # we use a contract multiplier to make sure we are correctly calculating
    # exposure as price * multiplier
    future_contract_multiplier = 2

    # this is the expected exposure for a position of one contract
    future_constant_exposure = (
        FUTURE_MINUTE_CONSTANT_CLOSE * future_contract_multiplier
    )

    @classmethod
    def make_futures_info(cls):
        return make_commodity_future_info(
            first_sid=ord('Z'),
            root_symbols=['Z'],
            years=[cls.START_DATE.year],
            multiplier=cls.future_contract_multiplier,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(TestConstantPrice, cls).init_class_fixtures()

        cls.equity = cls.asset_finder.retrieve_asset(
            cls.asset_finder.equities_sids[0],
        )
        cls.future = cls.asset_finder.retrieve_asset(
            cls.asset_finder.futures_sids[0],
        )

        cls.trading_minutes = pd.Index(
            cls.trading_calendar.minutes_for_sessions_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )
        cls.closes = pd.Index(
            cls.trading_calendar.session_closes_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )
        cls.closes.name = None

    def test_nop(self):
        perf = self.run_algorithm()

        zeros = pd.Series(0.0, index=self.closes)
        all_zero_fields = [
            'algorithm_period_return',
            'benchmark_period_return',
            'benchmark_returns',
            'capital_used',
            'downside_risk',
            'excess_return',
            'long_exposure',
            'long_value',
            'longs_count',
            'max_drawdown',
            'max_leverage',
            'short_exposure',
            'short_value',
            'shorts_count',
            'treasury_period_return',
        ]

        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        nan_then_zero = pd.Series(0.0, index=self.closes)
        nan_then_zero[0] = float('nan')
        nan_then_zero_fields = (
            'algo_volatility',
            'algorithm_volatility',
            'benchmark_volatility',
        )
        for field in nan_then_zero_fields:
            assert_equal(
                perf[field],
                nan_then_zero,
                check_names=False,
                msg=field,
            )

        empty_lists = pd.Series([[]] * len(self.closes), self.closes)
        empty_list_fields = (
            'orders',
            'positions',
            'transactions',
        )
        for field in empty_list_fields:
            assert_equal(
                perf[field],
                empty_lists,
                check_names=False,
                msg=field,
            )

    @parameter_space(
        direction=['long', 'short'],
        # checking the portfolio forces a sync; we want to ensure that the
        # perf packets are correct even without explicitly requesting the
        # portfolio every day. we also want to test that ``context.portfolio``
        # produces the expected values when queried mid-simulation
        check_portfolio_during_simulation=[True, False],
    )
    def test_equity_single_position(self,
                                    direction,
                                    check_portfolio_during_simulation):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        shares = 1 if direction == 'long' else -1

        def initialize(context):
            api.set_benchmark(self.equity)

            api.set_slippage(api.slippage.NoSlippage())
            api.set_commission(api.commission.NoCommission())

            context.first_bar = True

        if check_portfolio_during_simulation:
            portfolio_snapshots = {}

            def check_portfolio(portfolio, first_bar):
                portfolio_snapshots[api.get_datetime()] = portfolio_snapshot(
                    portfolio,
                )

                positions = portfolio.positions
                if first_bar:
                    assert_equal(positions, {})
                    return

                assert_equal(positions.keys(), [self.equity])
                position = positions[self.equity]
                assert_equal(position.last_sale_date, api.get_datetime())
                assert_equal(position.amount, shares)
                assert_equal(
                    position.last_sale_price,
                    self.EQUITY_MINUTE_CONSTANT_CLOSE,
                )
                assert_equal(position.asset, self.equity)
                assert_equal(
                    position.cost_basis,
                    self.EQUITY_MINUTE_CONSTANT_CLOSE,
                )
        else:
            def check_portfolio(portfolio, first_bar):
                pass

        def handle_data(context, data):
            first_bar = context.first_bar
            if first_bar:
                api.order(self.equity, shares)
                context.first_bar = False

            # take the snapshot after the order; ordering does not affect
            # the portfolio on the bar of the order, only the following bars
            check_portfolio(context.portfolio, first_bar)

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        zeros = pd.Series(0.0, index=self.closes)
        all_zero_fields = [
            'algorithm_period_return',
            'benchmark_period_return',
            'benchmark_returns',
            'downside_risk',
            'excess_return',
            'max_drawdown',
            'treasury_period_return',
        ]
        if direction == 'long':
            all_zero_fields.extend((
                'short_value',
                'shorts_count',
            ))
        else:
            all_zero_fields.extend((
                'long_value',
                'longs_count',
            ))
        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        ones = pd.Series(1, index=self.closes)
        if direction == 'long':
            count_field = 'longs_count'
        else:
            count_field = 'shorts_count'

        assert_equal(
            perf[count_field],
            ones,
            check_names=False,
            msg=field,
        )

        if direction == 'long':
            expected_exposure = pd.Series(
                self.EQUITY_MINUTE_CONSTANT_CLOSE,
                index=self.closes,
            )
            for field in 'long_value', 'long_exposure':
                assert_equal(
                    perf[field],
                    expected_exposure,
                    check_names=False,
                )
        else:
            expected_exposure = pd.Series(
                -self.EQUITY_MINUTE_CONSTANT_CLOSE,
                index=self.closes,
            )
            for field in 'short_value', 'short_exposure':
                assert_equal(
                    perf[field],
                    expected_exposure,
                    check_names=False,
                )

        nan_then_zero = pd.Series(0.0, index=self.closes)
        nan_then_zero[0] = float('nan')
        nan_then_zero_fields = (
            'algo_volatility',
            'algorithm_volatility',
            'benchmark_volatility',
        )
        for field in nan_then_zero_fields:
            assert_equal(
                perf[field],
                nan_then_zero,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        capital_base_series = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.closes,
        )

        # with no commissions, slippage, or returns our portfolio value stays
        # constant (at the capital base)
        assert_equal(
            perf['portfolio_value'],
            capital_base_series,
            check_names=False,
        )

        # leverage is gross market exposure / current notional capital
        # gross market exposure is
        # sum(long_exposure) + sum(abs(short_exposure))
        # current notional capital is the current portfolio value
        expected_max_leverage = (
            # we are exposed to only one share, the portfolio value is the
            # capital_base because we have no commissions, slippage, or
            # returns
            self.EQUITY_MINUTE_CONSTANT_CLOSE / capital_base_series
        )
        assert_equal(
            perf['max_leverage'],
            expected_max_leverage,
            check_names=False,
        )

        expected_cash = capital_base_series.copy()
        if direction == 'long':
            # we purchased one share on the first day
            cash_modifier = -self.EQUITY_MINUTE_CONSTANT_CLOSE
        else:
            # we sold one share on the first day
            cash_modifier = +self.EQUITY_MINUTE_CONSTANT_CLOSE

        expected_cash[1:] += cash_modifier

        assert_equal(
            perf['starting_cash'],
            expected_cash,
            check_names=False,
        )

        expected_cash[0] += cash_modifier
        assert_equal(
            perf['ending_cash'],
            expected_cash,
            check_names=False,
        )

        # we purchased one share on the first day
        expected_capital_used = pd.Series(0.0, index=self.closes)
        expected_capital_used[0] += cash_modifier

        assert_equal(
            perf['capital_used'],
            expected_capital_used,
            check_names=False,
        )

        # we hold one share so our positions exposure is that one share's price
        expected_position_exposure = pd.Series(
            -cash_modifier,
            index=self.closes,
        )
        for field in 'ending_value', 'ending_exposure':
            # for equities, position value and position exposure are the same
            assert_equal(
                perf[field],
                expected_position_exposure,
                check_names=False,
                msg=field,
            )

        # we don't start with any positions; the first day has no starting
        # exposure
        expected_position_exposure[0] = 0
        for field in 'starting_value', 'starting_exposure':
            # for equities, position value and position exposure are the same
            assert_equal(
                perf[field],
                expected_position_exposure,
                check_names=False,
                msg=field,
            )

        assert_equal(
            perf['trading_days'],
            pd.Series(
                np.arange(len(self.closes)) + 1,
                index=self.closes,
            ),
            check_names=False,
        )

        all_nan = pd.Series(np.nan, index=self.closes)
        all_nan_fields = (
            'alpha',
            'beta',
            'sortino',
        )
        for field in all_nan_fields:
            assert_equal(
                perf[field],
                all_nan,
                check_names=False,
                msg=field,
            )

        orders = perf['orders']

        expected_single_order = {
            'amount': shares,
            'commission': 0.0,
            'created': T('2014-01-06 14:31'),
            'dt': T('2014-01-06 14:32'),
            'filled': shares,
            'id': wildcard,
            'limit': None,
            'limit_reached': False,
            'reason': None,
            'sid': self.equity,
            'status': 1,
            'stop': None,
            'stop_reached': False
        }

        # we only order on the first day
        expected_orders = (
            [[expected_single_order]] +
            [[]] * (len(self.closes) - 1)
        )

        assert_equal(
            orders.tolist(),
            expected_orders,
            check_names=False,
        )
        assert_equal(
            orders.index,
            self.closes,
            check_names=False,
        )

        transactions = perf['transactions']

        expected_single_transaction = {
            'amount': shares,
            'commission': None,
            'dt': T('2014-01-06 14:32'),
            'order_id': wildcard,
            'price': 1.0,
            'sid': self.equity,
        }

        # since we only order on the first day, we should only transact on the
        # first day
        expected_transactions = (
            [[expected_single_transaction]] +
            [[]] * (len(self.closes) - 1)
        )

        assert_equal(
            transactions.tolist(),
            expected_transactions,
        )
        assert_equal(
            transactions.index,
            self.closes,
            check_names=False,
        )

        if not check_portfolio_during_simulation:
            return

        portfolio_snapshots = pd.DataFrame.from_dict(
            portfolio_snapshots,
            orient='index',
        )

        expected_starting_cash = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.trading_minutes,
        )
        assert_equal(
            portfolio_snapshots['starting_cash'],
            expected_starting_cash,
            check_names=False,
        )

        expected_portfolio_capital_used = pd.Series(
            cash_modifier,
            index=self.trading_minutes,
        )
        expected_portfolio_capital_used[0] = 0.0
        expected_capital_used[0] = 0
        assert_equal(
            portfolio_snapshots['capital_used'],
            expected_portfolio_capital_used,
            check_names=False,
        )

        zero_minutes = pd.Series(0.0, index=self.trading_minutes)
        for field in 'pnl', 'returns':
            assert_equal(
                portfolio_snapshots['pnl'],
                zero_minutes,
                check_names=False,
                msg=field,
            )

        reindex_columns = sorted(
            set(portfolio_snapshots.columns) - {
                'starting_cash',
                'capital_used',
                'pnl',
                'returns',
                'positions',
            },
        )
        minute_reindex = perf[reindex_columns].reindex(
            self.trading_minutes,
            method='bfill',
        )

        first_minute = self.trading_minutes[0]
        # the first minute should have the default values because we haven't
        # done anything yet
        minute_reindex.loc[first_minute, 'ending_cash'] = (
            self.SIM_PARAMS_CAPITAL_BASE
        )
        minute_reindex.loc[
            first_minute,
            ['ending_exposure', 'ending_value'],
        ] = 0

        assert_equal(
            portfolio_snapshots[reindex_columns],
            minute_reindex,
            check_names=False,
        )

    @parameter_space(
        direction=['long', 'short'],
        # checking the portfolio forces a sync; we want to ensure that the
        # perf packets are correct even without explicitly requesting the
        # portfolio every day. we also want to test that ``context.portfolio``
        # produces the expected values when queried mid-simulation
        check_portfolio_during_simulation=[True, False],
    )
    def test_future_single_position(self,
                                    direction,
                                    check_portfolio_during_simulation):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        if direction == 'long':
            contracts = 1
            expected_exposure = self.future_constant_exposure
        else:
            contracts = -1
            expected_exposure = -self.future_constant_exposure

        def initialize(context):
            api.set_benchmark(self.equity)

            api.set_slippage(us_futures=api.slippage.NoSlippage())
            api.set_commission(us_futures=api.commission.NoCommission())

            context.first_bar = True

        if check_portfolio_during_simulation:
            portfolio_snapshots = {}

            def check_portfolio(portfolio, first_bar):
                portfolio_snapshots[api.get_datetime()] = portfolio_snapshot(
                    portfolio,
                )

                positions = portfolio.positions
                if first_bar:
                    assert_equal(positions, {})
                    return

                assert_equal(positions.keys(), [self.future])
                position = positions[self.future]
                assert_equal(position.last_sale_date, api.get_datetime())
                assert_equal(position.amount, contracts)
                assert_equal(
                    position.last_sale_price,
                    self.FUTURE_MINUTE_CONSTANT_CLOSE,
                )
                assert_equal(position.asset, self.future)
                assert_equal(
                    position.cost_basis,
                    self.FUTURE_MINUTE_CONSTANT_CLOSE,
                )
        else:
            def check_portfolio(portfolio, first_bar):
                pass

        def handle_data(context, data):
            first_bar = context.first_bar
            if first_bar:
                api.order(self.future, contracts)
                context.first_bar = False

            # take the snapshot after the order; ordering does not affect
            # the portfolio on the bar of the order, only the following bars
            check_portfolio(context.portfolio, first_bar)

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        zeros = pd.Series(0.0, index=self.closes)
        all_zero_fields = [
            'algorithm_period_return',
            'benchmark_period_return',
            'benchmark_returns',
            'downside_risk',
            'excess_return',
            'max_drawdown',
            'treasury_period_return',

            # futures contracts have no value, just exposure
            'starting_value',
            'ending_value',
            'long_value',
            'short_value',
        ]
        if direction == 'long':
            all_zero_fields.extend((
                'short_value',
                'shorts_count',
            ))
        else:
            all_zero_fields.extend((
                'long_value',
                'longs_count',
            ))

        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        ones = pd.Series(1, index=self.closes)
        count_field = direction + 's_count'
        assert_equal(
            perf[count_field],
            ones,
            check_names=False,
            msg=count_field,
        )

        expected_exposure_series = pd.Series(
            expected_exposure,
            index=self.closes,
        )
        exposure_field = direction + '_exposure'
        assert_equal(
            perf[exposure_field],
            expected_exposure_series,
            check_names=False,
            msg=exposure_field,
        )

        nan_then_zero = pd.Series(0.0, index=self.closes)
        nan_then_zero[0] = float('nan')
        nan_then_zero_fields = (
            'algo_volatility',
            'algorithm_volatility',
            'benchmark_volatility',
        )
        for field in nan_then_zero_fields:
            assert_equal(
                perf[field],
                nan_then_zero,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        # with no commissions, entering or exiting a future position does not
        # affect your cash
        capital_base_series = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.closes,
        )

        # leverage is gross market exposure / current notional capital
        # gross market exposure is
        # sum(long_exposure) + sum(abs(short_exposure))
        # current notional capital is the current portfolio value
        expected_max_leverage = (
            self.future_constant_exposure / capital_base_series
        )
        assert_equal(
            perf['max_leverage'],
            expected_max_leverage,
            check_names=False,
        )

        # with no commissions, slippage, or returns our portfolio value stays
        # constant (at the capital base)
        for field in 'starting_cash', 'ending_cash', 'portfolio_value':
            assert_equal(
                perf[field],
                capital_base_series,
                check_names=False,
                msg=field,
            )

        # with no commissions, entering or exiting a future position does not
        # affect your cash; thus no capital gets used
        expected_capital_used = pd.Series(0.0, index=self.closes)

        assert_equal(
            perf['capital_used'],
            expected_capital_used,
            check_names=False,
        )

        # we hold one contract so our positions exposure is that one
        # contract's price
        expected_position_exposure = pd.Series(
            expected_exposure,
            index=self.closes,
        )
        assert_equal(
            perf['ending_exposure'],
            expected_position_exposure,
            check_names=False,
            check_dtype=False,
        )

        # we don't start with any positions; the first day has no starting
        # exposure
        expected_position_exposure[0] = 0
        assert_equal(
            perf['starting_exposure'],
            expected_position_exposure,
            check_names=False,
        )

        assert_equal(
            perf['trading_days'],
            pd.Series(
                np.arange(len(self.closes)) + 1,
                index=self.closes,
            ),
            check_names=False,
        )

        all_nan = pd.Series(np.nan, index=self.closes)
        all_nan_fields = (
            'alpha',
            'beta',
            'sortino',
        )
        for field in all_nan_fields:
            assert_equal(
                perf[field],
                all_nan,
                check_names=False,
                msg=field,
            )

        orders = perf['orders']

        # we only order on the first day
        expected_orders = [
            [{
                'amount': contracts,
                'commission': 0.0,
                'created': T('2014-01-06 14:31'),
                'dt': T('2014-01-06 14:32'),
                'filled': contracts,
                'id': wildcard,
                'limit': None,
                'limit_reached': False,
                'reason': None,
                'sid': self.future,
                'status': 1,
                'stop': None,
                'stop_reached': False
            }],
        ] + [[]] * (len(self.closes) - 1)

        assert_equal(
            orders.tolist(),
            expected_orders,
            check_names=False,
        )
        assert_equal(
            orders.index,
            self.closes,
            check_names=False,
        )

        transactions = perf['transactions']

        # since we only order on the first day, we should only transact on the
        # first day
        expected_transactions = [
            [{
                'amount': contracts,
                'commission': None,
                'dt': T('2014-01-06 14:32'),
                'order_id': wildcard,
                'price': 1.0,
                'sid': self.future,
            }],
        ] + [[]] * (len(self.closes) - 1)

        assert_equal(
            transactions.tolist(),
            expected_transactions,
            check_names=False,
        )
        assert_equal(
            transactions.index,
            self.closes,
            check_names=False,
        )

        if not check_portfolio_during_simulation:
            return

        portfolio_snapshots = pd.DataFrame.from_dict(
            portfolio_snapshots,
            orient='index',
        )

        expected_starting_cash = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.trading_minutes,
        )
        assert_equal(
            portfolio_snapshots['starting_cash'],
            expected_starting_cash,
            check_names=False,
        )

        zero_minutes = pd.Series(0.0, index=self.trading_minutes)
        for field in 'pnl', 'returns', 'capital_used':
            assert_equal(
                portfolio_snapshots['pnl'],
                zero_minutes,
                check_names=False,
                msg=field,
            )

        reindex_columns = sorted(
            set(portfolio_snapshots.columns) - {
                'starting_cash',
                'capital_used',
                'pnl',
                'returns',
                'positions',
            },
        )
        minute_reindex = perf[reindex_columns].reindex(
            self.trading_minutes,
            method='bfill',
        )

        first_minute = self.trading_minutes[0]
        # the first minute should have the default values because we haven't
        # done anything yet
        minute_reindex.loc[first_minute, 'ending_cash'] = (
            self.SIM_PARAMS_CAPITAL_BASE
        )
        minute_reindex.loc[
            first_minute,
            ['ending_exposure', 'ending_value'],
        ] = 0

        assert_equal(
            portfolio_snapshots[reindex_columns],
            minute_reindex,
            check_names=False,
        )


class TestFixedReturns(WithMakeAlgo, WithWerror, ZiplineTestCase):
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = True
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE = True

    START_DATE = T('2014-01-06')
    END_DATE = T('2014-01-10')

    # note: class attributes after this do not configure fixtures, they are
    # just used in this test suite

    # we use a contract multiplier to make sure we are correctly calculating
    # exposure as price * multiplier
    future_contract_multiplier = 2

    asset_start_price = 100
    asset_daily_returns = np.array([
        +0.02,  # up 2%
        -0.02,  # down 2%, this should give us less value that we started with
        +0.00,  # no returns
        +0.04,  # up 4%
    ])
    asset_daily_close = prices_generating_returns(
        asset_daily_returns,
        asset_start_price,
    )
    asset_daily_volume = 100000

    @classmethod
    def init_class_fixtures(cls):
        super(TestFixedReturns, cls).init_class_fixtures()

        cls.equity = cls.asset_finder.retrieve_asset(
            cls.asset_finder.equities_sids[0],
        )
        cls.future = cls.asset_finder.retrieve_asset(
            cls.asset_finder.futures_sids[0],
        )

        cls.trading_minutes = pd.Index(
            cls.trading_calendar.minutes_for_sessions_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )
        cls.closes = pd.Index(
            cls.trading_calendar.session_closes_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )
        cls.closes.name = None

    @classmethod
    def make_futures_info(cls):
        return make_commodity_future_info(
            first_sid=ord('Z'),
            root_symbols=['Z'],
            years=[cls.START_DATE.year],
            multiplier=cls.future_contract_multiplier,
        )

    @classmethod
    def _make_minute_bar_data(cls, calendar, sids):
        daily_close = cls.asset_daily_close
        daily_open = daily_close - 1
        daily_high = daily_close + 1
        daily_low = daily_close - 2
        random_state = np.random.RandomState(seed=1337)

        data = pd.concat(
            [
                simulate_minutes_for_day(
                    o,
                    h,
                    l,
                    c,
                    cls.asset_daily_volume,
                    trading_minutes=len(calendar.minutes_for_session(session)),
                    random_state=random_state,
                )
                for o, h, l, c, session in zip(
                    daily_open,
                    daily_high,
                    daily_low,
                    daily_close,
                    calendar.sessions_in_range(cls.START_DATE, cls.END_DATE),
                )
            ],
            ignore_index=True,
        )
        data.index = calendar.minutes_for_sessions_in_range(
            cls.START_DATE,
            cls.END_DATE,
        )

        for sid in sids:
            yield sid, data

    @classmethod
    def make_equity_minute_bar_data(cls):
        return cls._make_minute_bar_data(
            cls.trading_calendars[Equity],
            cls.asset_finder.equities_sids,
        )

    @classmethod
    def make_future_minute_bar_data(cls):
        return cls._make_minute_bar_data(
            cls.trading_calendars[Future],
            cls.asset_finder.futures_sids,
        )

    @parameter_space(
        direction=['long', 'short'],
        # checking the portfolio forces a sync; we want to ensure that the
        # perf packets are correct even without explicitly requesting the
        # portfolio every day. we also want to test that ``context.portfolio``
        # produces the expected values when queried mid-simulation
        check_portfolio_during_simulation=[True, False],
    )
    def test_equity_single_position(self,
                                    direction,
                                    check_portfolio_during_simulation):
        if direction not in ('long', 'short'):
            raise ValueError(
                'direction must be either long or short, got: %r' % direction,
            )

        shares = 1 if direction == 'long' else -1

        expected_fill_price = self.data_portal.get_scalar_asset_spot_value(
            self.equity,
            'close',
            # we expect to kill in the second bar of the first day
            self.trading_minutes[1],
            'minute',
        )

        def initialize(context):
            api.set_benchmark(self.equity)

            api.set_slippage(api.slippage.NoSlippage())
            api.set_commission(api.commission.NoCommission())

            context.first_bar = True

        if check_portfolio_during_simulation:
            portfolio_snapshots = {}

            def check_portfolio(data, portfolio, first_bar):
                portfolio_snapshots[api.get_datetime()] = portfolio_snapshot(
                    portfolio,
                )

                positions = portfolio.positions
                if first_bar:
                    assert_equal(positions, {})
                    return

                assert_equal(positions.keys(), [self.equity])
                position = positions[self.equity]
                assert_equal(position.last_sale_date, api.get_datetime())
                assert_equal(position.amount, shares)
                assert_equal(
                    position.last_sale_price,
                    data.current(self.equity, 'close'),
                )
                assert_equal(position.asset, self.equity)
                assert_equal(
                    position.cost_basis,
                    expected_fill_price,
                )
        else:
            def check_portfolio(data, portfolio, first_bar):
                pass

        def handle_data(context, data):
            first_bar = context.first_bar
            if first_bar:
                api.order(self.equity, shares)
                context.first_bar = False

            # take the snapshot after the order; ordering does not affect
            # the portfolio on the bar of the order, only the following bars
            check_portfolio(data, context.portfolio, first_bar)

        perf = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        zeros = pd.Series(0.0, index=self.closes)
        all_zero_fields = [
            'excess_return',
            'treasury_period_return',
        ]
        if direction == 'long':
            all_zero_fields.extend((
                'short_value',
                'shorts_count',
            ))
        else:
            all_zero_fields.extend((
                'long_value',
                'longs_count',
            ))
        for field in all_zero_fields:
            assert_equal(
                perf[field],
                zeros,
                check_names=False,
                check_dtype=False,
                msg=field,
            )

        ones = pd.Series(1, index=self.closes)
        if direction == 'long':
            count_field = 'longs_count'
        else:
            count_field = 'shorts_count'

        assert_equal(
            perf[count_field],
            ones,
            check_names=False,
            msg=field,
        )

        if direction == 'long':
            expected_exposure = pd.Series(
                self.asset_daily_close,
                index=self.closes,
            )
            exposure_fields = 'long_value', 'long_exposure'
        else:
            expected_exposure = pd.Series(
                -self.asset_daily_close,
                index=self.closes,
            )
            exposure_fields = 'short_value', 'short_exposure'

        for field in exposure_fields:
            assert_equal(
                perf[field],
                expected_exposure,
                check_names=False,
                msg=field,
            )

        if direction == 'long':
            delta = expected_fill_price
        else:
            delta = -expected_fill_price
        expected_portfolio_value = pd.Series(
            (
                self.SIM_PARAMS_CAPITAL_BASE +
                self.asset_daily_close -
                delta
            ),
            index=self.closes,
        )
        assert_equal(
            perf['portfolio_value'],
            expected_portfolio_value,
            check_names=False,
        )

        capital_base_series = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.closes,
        )

        # leverage is gross market exposure / current notional capital
        # gross market exposure is
        # sum(long_exposure) + sum(abs(short_exposure))
        # current notional capital is the current portfolio value
        expected_max_leverage = np.maximum.accumulate(
            expected_exposure / expected_portfolio_value,
        )
        assert_equal(
            perf['max_leverage'],
            expected_max_leverage,
            check_names=False,
        )

        expected_cash = capital_base_series.copy()
        if direction == 'long':
            # we purchased one share on the first day
            cash_modifier = -expected_fill_price
        else:
            # we sold one share on the first day
            cash_modifier = +expected_fill_price

        expected_cash[1:] += cash_modifier

        assert_equal(
            perf['starting_cash'],
            expected_cash,
            check_names=False,
        )

        expected_cash[0] += cash_modifier
        assert_equal(
            perf['ending_cash'],
            expected_cash,
            check_names=False,
        )

        # we purchased one share on the first day
        expected_capital_used = pd.Series(0.0, index=self.closes)
        expected_capital_used[0] += cash_modifier

        assert_equal(
            perf['capital_used'],
            expected_capital_used,
            check_names=False,
        )

        for field in 'ending_value', 'ending_exposure':
            # for equities, position value and position exposure are the same
            assert_equal(
                perf[field],
                expected_exposure,
                check_names=False,
                msg=field,
            )

        # we don't start with any positions; the first day has no starting
        # exposure
        expected_starting_exposure = expected_exposure.shift(1)
        expected_starting_exposure[0] = 0.0
        for field in 'starting_value', 'starting_exposure':
            # for equities, position value and position exposure are the same
            assert_equal(
                perf[field],
                expected_starting_exposure,
                check_names=False,
                msg=field,
            )

        assert_equal(
            perf['trading_days'],
            pd.Series(
                np.arange(len(self.closes)) + 1,
                index=self.closes,
            ),
            check_names=False,
        )

        orders = perf['orders']

        expected_single_order = {
            'amount': shares,
            'commission': 0.0,
            'created': T('2014-01-06 14:31'),
            'dt': T('2014-01-06 14:32'),
            'filled': shares,
            'id': wildcard,
            'limit': None,
            'limit_reached': False,
            'reason': None,
            'sid': self.equity,
            'status': 1,
            'stop': None,
            'stop_reached': False
        }

        # we only order on the first day
        expected_orders = (
            [[expected_single_order]] +
            [[]] * (len(self.closes) - 1)
        )

        assert_equal(
            orders.tolist(),
            expected_orders,
            check_names=False,
        )
        assert_equal(
            orders.index,
            self.closes,
            check_names=False,
        )

        transactions = perf['transactions']

        expected_single_transaction = {
            'amount': shares,
            'commission': None,
            'dt': T('2014-01-06 14:32'),
            'order_id': wildcard,
            'price': 1.0,
            'sid': self.equity,
        }

        # since we only order on the first day, we should only transact on the
        # first day
        expected_transactions = (
            [[expected_single_transaction]] +
            [[]] * (len(self.closes) - 1)
        )

        assert_equal(
            transactions.tolist(),
            expected_transactions,
        )
        assert_equal(
            transactions.index,
            self.closes,
            check_names=False,
        )

        if not check_portfolio_during_simulation:
            return

        portfolio_snapshots = pd.DataFrame.from_dict(
            portfolio_snapshots,
            orient='index',
        )

        expected_starting_cash = pd.Series(
            self.SIM_PARAMS_CAPITAL_BASE,
            index=self.trading_minutes,
        )
        assert_equal(
            portfolio_snapshots['starting_cash'],
            expected_starting_cash,
            check_names=False,
        )

        expected_portfolio_capital_used = pd.Series(
            cash_modifier,
            index=self.trading_minutes,
        )
        expected_portfolio_capital_used[0] = 0.0
        expected_capital_used[0] = 0
        assert_equal(
            portfolio_snapshots['capital_used'],
            expected_portfolio_capital_used,
            check_names=False,
        )

        zero_minutes = pd.Series(0.0, index=self.trading_minutes)
        for field in 'pnl', 'returns':
            assert_equal(
                portfolio_snapshots['pnl'],
                zero_minutes,
                check_names=False,
                msg=field,
            )

        reindex_columns = sorted(
            set(portfolio_snapshots.columns) - {
                'starting_cash',
                'capital_used',
                'pnl',
                'returns',
                'positions',
            },
        )
        minute_reindex = perf[reindex_columns].reindex(
            self.trading_minutes,
            method='bfill',
        )
        # the first minute should have the default values because we haven't
        # done anything yet
        minute_reindex['ending_cash'].iloc[0] = self.SIM_PARAMS_CAPITAL_BASE
        minute_reindex['ending_exposure'].iloc[0] = 0
        minute_reindex['ending_value'].iloc[0] = 0

        assert_equal(
            portfolio_snapshots[reindex_columns],
            minute_reindex,
            check_names=False,
        )
