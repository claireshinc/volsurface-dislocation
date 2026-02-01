#!/usr/bin/env python3
"""Daily options data collector and SVI surface fitter.

Fetches options chain data from OpenBB, fits SVI surfaces, extracts features,
runs anomaly detection, and stores results in the database.

Scheduled to run daily at 4:30 PM ET (after market close).

Usage:
    # Run once immediately
    python -m scripts.daily_collector --once

    # Run once for specific tickers
    python -m scripts.daily_collector --once --tickers SPY QQQ IWM

    # Run as daemon (scheduled daily at 4:30 PM ET)
    python -m scripts.daily_collector --daemon

    # Run as daemon with custom time
    python -m scripts.daily_collector --daemon --time 16:45
"""

import argparse
import logging
import signal
import sys
from datetime import datetime, date
from typing import Any
import json

import pytz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("daily_collector.log"),
    ],
)
logger = logging.getLogger(__name__)

# Default tickers to collect
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]

# Timezone for scheduling
ET = pytz.timezone("America/New_York")


class DailyCollector:
    """Collects options data and fits SVI surfaces daily.

    Example:
        >>> collector = DailyCollector(tickers=["SPY", "QQQ"])
        >>> collector.run_collection()
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        database_url: str | None = None,
        rate: float = 0.05,
        dividend_yield: float = 0.0,
    ):
        """Initialize daily collector.

        Args:
            tickers: List of tickers to collect (default: major ETFs and stocks)
            database_url: Database URL (default: from settings)
            rate: Risk-free rate for SVI fitting
            dividend_yield: Dividend yield for SVI fitting
        """
        self.tickers = tickers or DEFAULT_TICKERS
        self.database_url = database_url
        self.rate = rate
        self.dividend_yield = dividend_yield

        # Lazy load components
        self._db_initialized = False
        self._detector = None

    def _init_db(self) -> None:
        """Initialize database connection."""
        if self._db_initialized:
            return

        from data.database import init_database
        init_database(self.database_url)
        self._db_initialized = True
        logger.info("Database initialized")

    def _get_detector(self):
        """Get or create anomaly detector with loaded history."""
        if self._detector is None:
            from analytics.anomaly_detector import AnomalyDetector
            self._detector = AnomalyDetector()
        return self._detector

    def _load_historical_features(self, ticker: str) -> None:
        """Load historical features into anomaly detector from database."""
        from data.database import get_session, SurfaceFeaturesRecord

        detector = self._get_detector()

        with next(get_session()) as session:
            records = (
                session.query(SurfaceFeaturesRecord)
                .filter(SurfaceFeaturesRecord.ticker == ticker)
                .order_by(SurfaceFeaturesRecord.timestamp.desc())
                .limit(252)
                .all()
            )

            if not records:
                logger.info(f"No historical data for {ticker}")
                return

            # Load each feature's history
            for feature_name in detector.FEATURE_NAMES:
                timestamps = []
                values = []

                for record in records:
                    value = getattr(record, feature_name, None)
                    if value is not None:
                        timestamps.append(record.timestamp)
                        values.append(value)

                if timestamps:
                    detector.load_history(ticker, feature_name, timestamps, values)

            logger.info(f"Loaded {len(records)} historical records for {ticker}")

    def collect_ticker(self, ticker: str) -> dict[str, Any]:
        """Collect and process data for a single ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with collection results
        """
        from analytics.svi_fitter import SVIFitter
        from analytics.feature_extractor import FeatureExtractor
        from analytics.trade_ideas import TradeIdeaGenerator
        from data.collector import OptionsDataCollector
        from data.database import (
            get_session,
            OptionsChainRecord,
            FittedSurfaceRecord,
            SurfaceFeaturesRecord,
            AnomalyLogRecord,
            SurfaceRepository,
        )

        result = {
            "ticker": ticker,
            "timestamp": datetime.now(),
            "success": False,
            "error": None,
            "chain_size": 0,
            "expiries_fitted": 0,
            "anomalies_detected": 0,
            "trade_ideas": 0,
        }

        try:
            # Fetch options chain
            logger.info(f"Fetching options chain for {ticker}")
            collector = OptionsDataCollector(provider="cboe", fallback_provider="yfinance")
            chain = collector.get_chain(ticker, min_oi=10)

            result["chain_size"] = len(chain.chain_data)
            logger.info(f"Fetched {result['chain_size']} options for {ticker}")

            # Fit SVI surfaces
            fitter = SVIFitter()
            params_list = []

            for expiry in chain.expiries[:8]:  # Limit to 8 nearest expiries
                try:
                    k, w, T = collector.prepare_for_svi_fitting(
                        chain, expiry, self.rate, self.dividend_yield
                    )
                    params = fitter.fit(k, w, T)
                    params_list.append((expiry, params))
                    logger.debug(f"Fitted {ticker} {expiry}: RMSE={params.fit_error:.6f}")
                except Exception as e:
                    logger.warning(f"Could not fit {ticker} {expiry}: {e}")

            result["expiries_fitted"] = len(params_list)

            if not params_list:
                raise ValueError("Could not fit any expiries")

            # Extract features
            extractor = FeatureExtractor(chain.spot_price, self.rate, self.dividend_yield)
            features = extractor.extract(
                [p for _, p in params_list],
                ticker,
                timestamp=datetime.now(),
            )

            # Load historical data and run anomaly detection
            self._load_historical_features(ticker)
            detector = self._get_detector()
            report = detector.analyze(features)

            result["anomalies_detected"] = report.anomaly_count

            # Generate trade ideas
            generator = TradeIdeaGenerator()
            ideas = generator.generate(report)
            result["trade_ideas"] = len(ideas)

            # Store in database
            self._init_db()

            with next(get_session()) as session:
                repo = SurfaceRepository(session)

                # Store raw chain
                chain_record = OptionsChainRecord(
                    timestamp=datetime.now(),
                    ticker=ticker,
                    spot_price=chain.spot_price,
                )
                chain_record.set_chain_data(chain.to_dict())
                session.add(chain_record)

                # Store fitted surfaces
                for expiry, params in params_list:
                    days = (expiry - date.today()).days
                    surface_record = FittedSurfaceRecord(
                        timestamp=datetime.now(),
                        ticker=ticker,
                        expiry_days=days,
                        expiry_years=params.expiry_years,
                        svi_a=params.a,
                        svi_b=params.b,
                        svi_rho=params.rho,
                        svi_m=params.m,
                        svi_sigma=params.sigma,
                        fit_error=params.fit_error,
                    )
                    session.add(surface_record)

                # Store features
                repo.save_features(features)

                # Store anomalies
                for anomaly in report.anomalies:
                    if anomaly.is_anomaly:
                        # Find matching trade idea
                        suggestion = None
                        for idea in ideas:
                            if anomaly.feature_name in idea.related_anomalies:
                                suggestion = f"{idea.strategy}: {idea.description}"
                                break

                        repo.save_anomaly(ticker, anomaly, suggestion)

                session.commit()

            result["success"] = True
            logger.info(
                f"Completed {ticker}: {result['expiries_fitted']} surfaces, "
                f"{result['anomalies_detected']} anomalies, {result['trade_ideas']} ideas"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to collect {ticker}: {e}", exc_info=True)

        return result

    def run_collection(self) -> list[dict[str, Any]]:
        """Run collection for all tickers.

        Returns:
            List of results for each ticker
        """
        logger.info(f"Starting daily collection for {len(self.tickers)} tickers")
        start_time = datetime.now()

        results = []
        for ticker in self.tickers:
            result = self.collect_ticker(ticker)
            results.append(result)

            # Add detector observation for next run
            if result["success"]:
                detector = self._get_detector()
                # Features are already loaded during collect_ticker

        # Summary
        successful = sum(1 for r in results if r["success"])
        total_anomalies = sum(r["anomalies_detected"] for r in results)
        total_ideas = sum(r["trade_ideas"] for r in results)
        duration = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Collection complete: {successful}/{len(self.tickers)} successful, "
            f"{total_anomalies} anomalies, {total_ideas} trade ideas, "
            f"duration: {duration:.1f}s"
        )

        # Write summary to file
        summary = {
            "timestamp": datetime.now().isoformat(),
            "tickers_attempted": len(self.tickers),
            "tickers_successful": successful,
            "total_anomalies": total_anomalies,
            "total_trade_ideas": total_ideas,
            "duration_seconds": duration,
            "results": results,
        }

        summary_file = f"collection_summary_{date.today().isoformat()}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary written to {summary_file}")

        return results

    def generate_daily_report(self, results: list[dict[str, Any]]) -> str:
        """Generate a daily summary report.

        Args:
            results: Collection results

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            f"VOLATILITY ARCHAEOLOGIST - DAILY REPORT",
            f"Date: {date.today().isoformat()}",
            f"Time: {datetime.now().strftime('%H:%M:%S')} ET",
            "=" * 60,
            "",
        ]

        # Summary stats
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        lines.extend([
            f"Tickers Processed: {len(successful)}/{len(results)}",
            f"Total Anomalies: {sum(r['anomalies_detected'] for r in results)}",
            f"Trade Ideas Generated: {sum(r['trade_ideas'] for r in results)}",
            "",
        ])

        # Anomalies by ticker
        anomaly_tickers = [(r["ticker"], r["anomalies_detected"])
                          for r in successful if r["anomalies_detected"] > 0]

        if anomaly_tickers:
            lines.append("ANOMALIES DETECTED:")
            for ticker, count in sorted(anomaly_tickers, key=lambda x: -x[1]):
                lines.append(f"  {ticker}: {count} anomalies")
            lines.append("")

        # Failed tickers
        if failed:
            lines.append("FAILED TICKERS:")
            for r in failed:
                lines.append(f"  {r['ticker']}: {r['error']}")
            lines.append("")

        lines.append("=" * 60)

        report = "\n".join(lines)

        # Save report
        report_file = f"daily_report_{date.today().isoformat()}.txt"
        with open(report_file, "w") as f:
            f.write(report)

        return report


def run_scheduled_job(collector: DailyCollector) -> None:
    """Run the scheduled collection job."""
    logger.info("Scheduled job triggered")

    # Check if market was open today (skip weekends)
    today = datetime.now(ET)
    if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
        logger.info("Skipping collection - weekend")
        return

    results = collector.run_collection()
    report = collector.generate_daily_report(results)
    print(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Daily options data collector and SVI surface fitter"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once immediately and exit",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon with scheduled execution",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="16:30",
        help="Time to run daily (HH:MM in ET), default: 16:30",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Tickers to collect (default: major ETFs and stocks)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        help="Risk-free rate (default: 0.05)",
    )

    args = parser.parse_args()

    if not args.once and not args.daemon:
        parser.print_help()
        print("\nError: Must specify --once or --daemon")
        sys.exit(1)

    collector = DailyCollector(
        tickers=args.tickers,
        rate=args.rate,
    )

    if args.once:
        logger.info("Running one-time collection")
        results = collector.run_collection()
        report = collector.generate_daily_report(results)
        print(report)

    elif args.daemon:
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error("APScheduler not installed. Install with: pip install apscheduler")
            print("For daemon mode, install APScheduler: pip install apscheduler")
            sys.exit(1)

        # Parse time
        hour, minute = map(int, args.time.split(":"))

        logger.info(f"Starting daemon, scheduled daily at {args.time} ET")

        scheduler = BlockingScheduler(timezone=ET)

        # Schedule daily job
        scheduler.add_job(
            run_scheduled_job,
            CronTrigger(hour=hour, minute=minute, timezone=ET),
            args=[collector],
            id="daily_collection",
            name="Daily Options Collection",
            misfire_grace_time=3600,  # 1 hour grace period
        )

        # Handle shutdown gracefully
        def shutdown(signum, frame):
            logger.info("Shutdown signal received")
            scheduler.shutdown(wait=False)
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        print(f"Daemon started. Scheduled to run daily at {args.time} ET")
        print("Press Ctrl+C to stop")

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Daemon stopped")


if __name__ == "__main__":
    main()
