"""Main test runner for all Memory Reflection Service end-to-end tests."""

import asyncio
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Import all test modules
from test_full_pipeline import run_e2e_tests as run_pipeline_tests
from test_stress import run_stress_tests
from test_monitoring import run_monitoring_tests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestSuite:
    """Complete test suite for Memory Reflection Service."""
    
    def __init__(self):
        self.start_time = None
        self.results = []
    
    async def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run all end-to-end tests in sequence."""
        self.start_time = datetime.utcnow()
        
        logger.info("🧪 Memory Reflection Service - Complete Test Suite")
        logger.info("=" * 80)
        logger.info(f"Test suite started at: {self.start_time.isoformat()}")
        logger.info("=" * 80)
        
        test_suites = [
            {
                "name": "Pipeline Tests",
                "description": "End-to-end pipeline functionality", 
                "runner": run_pipeline_tests,
                "critical": True
            },
            {
                "name": "Stress Tests",
                "description": "Performance under load",
                "runner": run_stress_tests, 
                "critical": True
            },
            {
                "name": "Monitoring Tests",
                "description": "Health checks and monitoring",
                "runner": run_monitoring_tests,
                "critical": False
            }
        ]
        
        passed = 0
        failed = 0
        critical_failed = 0
        
        for i, test_suite in enumerate(test_suites, 1):
            logger.info(f"\n📋 RUNNING TEST SUITE {i}/{len(test_suites)}: {test_suite['name']}")
            logger.info(f"Description: {test_suite['description']}")
            logger.info("-" * 60)
            
            suite_start = time.time()
            
            try:
                success = await test_suite["runner"]()
                suite_time = time.time() - suite_start
                
                if success:
                    logger.info(f"✅ {test_suite['name']} PASSED ({suite_time:.1f}s)")
                    passed += 1
                    self.results.append({
                        "suite": test_suite["name"],
                        "status": "PASSED",
                        "duration": suite_time,
                        "critical": test_suite["critical"]
                    })
                else:
                    logger.error(f"❌ {test_suite['name']} FAILED ({suite_time:.1f}s)")
                    failed += 1
                    if test_suite["critical"]:
                        critical_failed += 1
                    self.results.append({
                        "suite": test_suite["name"],
                        "status": "FAILED",
                        "duration": suite_time,
                        "critical": test_suite["critical"]
                    })
                    
            except Exception as e:
                suite_time = time.time() - suite_start
                logger.error(f"💥 {test_suite['name']} CRASHED: {e} ({suite_time:.1f}s)")
                failed += 1
                if test_suite["critical"]:
                    critical_failed += 1
                self.results.append({
                    "suite": test_suite["name"],
                    "status": "CRASHED", 
                    "duration": suite_time,
                    "critical": test_suite["critical"],
                    "error": str(e)
                })
        
        end_time = datetime.utcnow()
        total_time = (end_time - self.start_time).total_seconds()
        
        # Generate final report
        return self._generate_final_report(passed, failed, critical_failed, total_time)
    
    def _generate_final_report(self, passed: int, failed: int, critical_failed: int, total_time: float) -> Dict[str, Any]:
        """Generate final test report."""
        total_tests = passed + failed
        success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info("🏁 MEMORY REFLECTION SERVICE TEST SUITE COMPLETE")
        logger.info("=" * 80)
        
        logger.info(f"📊 TEST SUMMARY:")
        logger.info(f"   • Total test suites: {total_tests}")
        logger.info(f"   • Passed: {passed}")
        logger.info(f"   • Failed: {failed}")
        logger.info(f"   • Critical failures: {critical_failed}")
        logger.info(f"   • Success rate: {success_rate:.1f}%")
        logger.info(f"   • Total duration: {total_time:.1f}s")
        
        logger.info(f"\n📋 DETAILED RESULTS:")
        for result in self.results:
            status_emoji = "✅" if result["status"] == "PASSED" else "❌" if result["status"] == "FAILED" else "💥"
            critical_marker = "🔴" if result.get("critical", False) else "🟡"
            logger.info(f"   {status_emoji} {critical_marker} {result['suite']}: {result['status']} ({result['duration']:.1f}s)")
            if "error" in result:
                logger.info(f"      Error: {result['error']}")
        
        # Overall assessment
        if critical_failed > 0:
            logger.error("❌ CRITICAL TEST FAILURES - System NOT ready for production")
            overall_status = "CRITICAL_FAILURE"
        elif failed > 0:
            logger.warning("⚠️ NON-CRITICAL TEST FAILURES - Review before production")
            overall_status = "WARNING"
        else:
            logger.info("🎉 ALL TESTS PASSED - System ready for production!")
            overall_status = "SUCCESS"
        
        logger.info("\n" + "=" * 80)
        
        return {
            "overall_status": overall_status,
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "critical_failed": critical_failed,
            "success_rate": success_rate,
            "total_duration": total_time,
            "results": self.results,
            "ready_for_production": critical_failed == 0,
            "timestamp": datetime.utcnow().isoformat()
        }

async def main():
    """Main entry point for test suite."""
    try:
        test_suite = TestSuite()
        final_report = await test_suite.run_complete_test_suite()
        
        # Return appropriate exit code
        if final_report["overall_status"] == "CRITICAL_FAILURE":
            return 2  # Critical failure
        elif final_report["overall_status"] == "WARNING":
            return 1  # Non-critical failures
        else:
            return 0  # Success
            
    except Exception as e:
        logger.error(f"💥 Test suite crashed: {e}")
        return 3  # Test suite failure

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)