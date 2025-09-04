.PHONY: test test-unit test-integration test-coverage test-watch install-test-deps

# Install test dependencies
install-test-deps:
	pip install -r requirements-test.txt

# Run all tests
test:
	pytest

# Run only unit tests
test-unit:
	pytest tests/test_main.py -m "not integration"

# Run only integration tests  
test-integration:
	pytest tests/test_integration.py -m integration

# Run tests with coverage report
test-coverage:
	pytest --cov=main --cov-report=html --cov-report=term

# Run tests in watch mode (requires pytest-watch)
test-watch:
	ptw --runner "pytest -v"

# Run specific test file
test-main:
	pytest tests/test_main.py -v

# Run tests with detailed output
test-verbose:
	pytest -v -s

# Clean test artifacts
clean-test:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete