"""
Test script to verify NaN fixes are working
Run this to validate the numerical stability improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_diagnostic():
    """Run the diagnostic script to test for NaN issues"""
    logger.info("Running NaN diagnostic script...")
    
    diagnostic_script = os.path.join(os.path.dirname(__file__), 'diagnose_nan_issues.py')
    
    try:
        result = subprocess.run(['python', diagnostic_script], 
                              capture_output=True, text=True, timeout=300)
        
        logger.info("Diagnostic output:")
        print(result.stdout)
        
        if result.stderr:
            logger.warning("Diagnostic warnings/errors:")
            print(result.stderr)
            
        success = result.returncode == 0
        logger.info(f"Diagnostic result: {'PASS' if success else 'FAIL'}")
        return success
        
    except subprocess.TimeoutExpired:
        logger.error("Diagnostic script timed out")
        return False
    except Exception as e:
        logger.error(f"Failed to run diagnostic: {e}")
        return False

def main():
    """Main test function"""
    logger.info("="*60)
    logger.info("TESTING NUMERICAL STABILITY FIXES")
    logger.info("="*60)
    
    logger.info("Applied fixes:")
    logger.info("1. Tighter gradient clipping (0.5 instead of 1.0)")
    logger.info("2. More conservative learning rate (0.0001 instead of 0.001)")
    logger.info("3. Reduced loss weights (alpha=1.0, beta=0.5)")
    logger.info("4. Enhanced KL loss bounds (-3 to 3 for mean, -6 to 1 for log_var)")
    logger.info("5. Improved clustering loss stability")
    logger.info("6. Huber loss for reconstruction (more robust)")
    logger.info("7. NaN gradient detection and skipping")
    logger.info("8. TerminateOnNaN callback")
    logger.info("")
    
    # Run the diagnostic
    success = run_diagnostic()
    
    if success:
        logger.info("✅ All tests passed! The fixes appear to be working.")
        logger.info("\nNext steps:")
        logger.info("1. Run training with: python main.py train")
        logger.info("2. Monitor the logs for any remaining NaN issues")
        logger.info("3. The training should now be much more stable")
    else:
        logger.error("❌ Some tests failed. Additional fixes may be needed.")
        logger.info("\nIf issues persist, consider:")
        logger.info("1. Further reducing learning rate")
        logger.info("2. Increasing gradient clipping strength")
        logger.info("3. Using mixed precision with loss scaling")
        
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)