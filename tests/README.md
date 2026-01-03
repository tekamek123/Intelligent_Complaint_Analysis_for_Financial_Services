# Test Suite

This directory contains unit tests for the complaint analysis project.

## Test Files

- `test_task1.py`: Tests for Task 1 (EDA and preprocessing)
- `test_task2.py`: Tests for Task 2 (Chunking, embedding, and vector store)
- `conftest.py`: Shared pytest fixtures and configuration

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_task1.py -v
pytest tests/test_task2.py -v
```

### Run with coverage
```bash
pytest tests/ -v --cov=notebooks --cov-report=term-missing
```

### Run specific test class
```bash
pytest tests/test_task1.py::TestCleanText -v
```

## Test Coverage

The test suite covers:

### Task 1 Tests
- Data loading error handling
- Text cleaning functionality
- Dataset filtering
- Product distribution analysis
- Narrative length analysis
- Data validation

### Task 2 Tests
- Column identification
- Stratified sampling
- Text chunking
- Embedding generation
- Vector store creation
- Error handling for invalid inputs

## Notes

- Some tests may be skipped if external dependencies (like embedding models) are not available
- Tests use pytest fixtures for sample data
- Mocking is used where appropriate to avoid requiring actual data files

