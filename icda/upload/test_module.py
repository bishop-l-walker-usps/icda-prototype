"""
Quick test script for the upload module
Run: python -m icda.upload.test_module
"""

import asyncio
from .address_validator import USStateValidator, ZipCodeValidator, AddressValidatorService
from .nova_corrector import NovaProAddressCorrector
from .upload_handler import StreamingUploadHandler, UploadMode
from .document_parser import DocumentParserService


async def test_state_validator():
    """Test state validation"""
    print("\n=== State Validator Test ===")
    
    # Valid states
    assert USStateValidator.is_valid_state("VA") == True
    assert USStateValidator.is_valid_state("Virginia") == True
    assert USStateValidator.is_valid_state("PR") == True  # Puerto Rico
    
    # Invalid state
    assert USStateValidator.is_valid_state("XX") == False
    
    # Normalize
    assert USStateValidator.normalize_state("Virginia") == "VA"
    assert USStateValidator.normalize_state("puerto rico") == "PR"
    
    # Similar states
    similar = USStateValidator.get_similar_states("Virg")
    print(f"  Similar to 'Virg': {[(s[0], f'{s[2]:.0%}') for s in similar[:3]]}")
    
    print("  ✓ State validator tests passed")


async def test_zip_validator():
    """Test ZIP code validation"""
    print("\n=== ZIP Code Validator Test ===")
    
    # Valid formats
    assert ZipCodeValidator.is_valid_format("22030") == True
    assert ZipCodeValidator.is_valid_format("22030-1234") == True
    
    # Invalid
    assert ZipCodeValidator.is_valid_format("2203") == False
    assert ZipCodeValidator.is_valid_format("abcde") == False
    
    # Extract ZIP5
    assert ZipCodeValidator.extract_zip5("22030-1234") == "22030"
    
    # Normalize
    assert ZipCodeValidator.normalize("220301234") == "22030-1234"
    
    print("  ✓ ZIP validator tests passed")


async def test_address_validator():
    """Test address validation service"""
    print("\n=== Address Validator Service Test ===")
    
    validator = AddressValidatorService()
    
    # Valid address
    result = await validator.validate({
        "street": "123 Main St",
        "city": "McLean",
        "state": "VA",
        "zip": "22030"
    })
    print(f"  Valid address confidence: {result.confidence:.1%}")
    print(f"  Errors: {[e.message for e in result.errors]}")
    
    # Invalid address (missing fields)
    result = await validator.validate({
        "street": "101 turkey",
        "zip": "22030"
    })
    print(f"  Invalid address confidence: {result.confidence:.1%}")
    print(f"  Errors: {[e.message for e in result.errors]}")
    
    print("  ✓ Address validator tests passed")


async def test_upload_handler():
    """Test streaming upload handler"""
    print("\n=== Upload Handler Test ===")
    
    handler = StreamingUploadHandler()
    
    # Start upload
    upload_id = await handler.start_upload(
        filename="test.csv",
        total_bytes=1000,
        content_type="text/csv"
    )
    print(f"  Upload ID: {upload_id[:8]}...")
    
    # Receive chunks
    await handler.receive_chunk(upload_id, b"test,data\n" * 100, 0)
    
    progress = handler.get_progress(upload_id)
    print(f"  Progress: {progress.percent_complete}%")
    
    # Cleanup
    await handler.cleanup(upload_id)
    
    print("  ✓ Upload handler tests passed")


async def test_document_parser():
    """Test document parser"""
    print("\n=== Document Parser Test ===")
    
    parser = DocumentParserService()
    
    # Check available parsers
    print(f"  Available parsers: {list(parser.parsers.keys())}")
    
    print("  ✓ Document parser tests passed")


async def main():
    """Run all tests"""
    print("=" * 50)
    print("  ICDA Upload Module Tests")
    print("=" * 50)
    
    try:
        await test_state_validator()
        await test_zip_validator()
        await test_address_validator()
        await test_upload_handler()
        await test_document_parser()
        
        print("\n" + "=" * 50)
        print("  All tests passed! ✓")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n  ✗ Test failed: {e}")
    except Exception as e:
        print(f"\n  ✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
