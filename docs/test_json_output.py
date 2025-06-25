#!/usr/bin/env python3
"""
Test script for the JSON output extension.

This script helps verify that:
1. JSON files are generated alongside HTML files
2. JSON structure contains expected fields
3. Content extraction is working properly
"""

import json
import os
from pathlib import Path


def test_json_output(build_dir: str = "_build/html"):
    """Test the JSON output extension results."""
    build_path = Path(build_dir)
    
    if not build_path.exists():
        print(f"❌ Build directory not found: {build_path}")
        print("Run 'make html' first to build the documentation.")
        return False
    
    print(f"🔍 Testing JSON output in: {build_path}")
    
    # Find all HTML files
    html_files = list(build_path.rglob("*.html"))
    json_files = list(build_path.rglob("*.json"))
    
    print(f"📄 Found {len(html_files)} HTML files")
    print(f"📋 Found {len(json_files)} JSON files")
    
    # Test specific JSON files
    test_cases = [
        "index.json",
        "about/index.json", 
        "get-started/index.json",
        "feature-set-a/category-a/topic-a/index.json"
    ]
    
    success_count = 0
    total_tests = 0
    
    for test_case in test_cases:
        json_path = build_path / test_case
        html_path = build_path / test_case.replace('.json', '.html')
        
        total_tests += 1
        
        print(f"\n🧪 Testing: {test_case}")
        
        # Check if JSON file exists
        if not json_path.exists():
            print(f"   ❌ JSON file missing: {json_path}")
            continue
            
        # Check if corresponding HTML exists
        if not html_path.exists():
            print(f"   ⚠️  HTML file missing: {html_path}")
        else:
            print(f"   ✅ HTML file exists: {html_path}")
        
        # Validate JSON structure
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required fields (optimized structure)
            required_fields = ['id', 'title', 'url', 'last_modified']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                print(f"   ❌ Missing required fields: {missing_fields}")
                continue
                
            print(f"   ✅ All required fields present")
            
            # Check optional fields (LLM/search optimized)
            optional_fields = ['content', 'format', 'summary', 'headings', 'tags', 'categories', 'description', 'children']
            present_optional = [field for field in optional_fields if field in data]
            print(f"   📋 Optional fields present: {present_optional}")
            
            # Validate content (prioritize markdown/text)
            if 'content' in data and data['content']:
                content_format = data.get('format', 'unknown')
                print(f"   ✅ Content extracted ({len(data['content'])} chars, format: {content_format})")
            
            if 'summary' in data and data['summary']:
                print(f"   ✅ Summary extracted ({len(data['summary'])} chars)")
                
            if 'headings' in data and isinstance(data['headings'], list):
                print(f"   ✅ Headings extracted ({len(data['headings'])} headings)")
            
            # Check metadata fields
            metadata_fields = ['tags', 'categories', 'description']
            present_metadata = [field for field in metadata_fields if field in data]
            if present_metadata:
                print(f"   📝 Metadata fields: {present_metadata}")
            
            # Validate children (for directory indexes)
            if 'children' in data and isinstance(data['children'], list):
                print(f"   ✅ Child documents included ({len(data['children'])} children)")
                
                # For main index, show total documents
                if 'total_documents' in data:
                    print(f"   🌍 Total documents in search index: {data['total_documents']}")
                
                # Validate first child structure if present
                if data['children']:
                    first_child = data['children'][0]
                    child_fields = ['id', 'title', 'url']
                    missing_child_fields = [field for field in child_fields if field not in first_child]
                    if missing_child_fields:
                        print(f"   ⚠️  Child missing fields: {missing_child_fields}")
                    else:
                        print(f"   ✅ Child document structure valid")
                        
                        # Check if child has content
                        if 'content' in first_child:
                            child_format = first_child.get('format', 'unknown')
                            print(f"   📄 Child content format: {child_format}")
            
            success_count += 1
            print(f"   ✅ JSON validation passed")
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Invalid JSON: {e}")
        except Exception as e:
            print(f"   ❌ Error reading JSON: {e}")
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"   ✅ Passed: {success_count}/{total_tests}")
    print(f"   ❌ Failed: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! JSON output extension is working correctly.")
        return True
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
        return False


def show_json_example(build_dir: str = "_build/html", filename: str = "index.json"):
    """Show an example of generated JSON output."""
    json_path = Path(build_dir) / filename
    
    if not json_path.exists():
        print(f"❌ JSON file not found: {json_path}")
        return
    
    print(f"📋 Example JSON output from {filename}:\n")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Pretty print with truncated content for readability
        display_data = data.copy()
        
        # Truncate long content fields
        for field in ['html', 'text']:
            if field in display_data and len(str(display_data[field])) > 200:
                display_data[field] = str(display_data[field])[:200] + "... [truncated]"
        
        print(json.dumps(display_data, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")


if __name__ == "__main__":
    import sys
    
    build_dir = sys.argv[1] if len(sys.argv) > 1 else "_build/html"
    
    print("🧪 Testing JSON Output Extension")
    print("=" * 50)
    
    success = test_json_output(build_dir)
    
    if success:
        print("\n" + "=" * 50)
        show_json_example(build_dir) 