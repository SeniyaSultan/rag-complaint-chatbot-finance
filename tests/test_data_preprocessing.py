"""
Comprehensive tests for data preprocessing module
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.data_preprocessing import DataPreprocessor
from src.utils.error_handling import DataValidationError

class TestDataPreprocessor:
    """Test suite for DataPreprocessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample complaint data for testing"""
        return pd.DataFrame({
            'complaint_id': range(1, 11),
            'Date received': pd.date_range('2024-01-01', periods=10, freq='D'),
            'Product': ['Credit card'] * 5 + ['Personal Loan'] * 3 + ['Mortgage'] * 2,
            'Consumer complaint narrative': [
                'I am writing to file a complaint about billing issues. My account was charged incorrectly.',
                'Issue with service quality. The customer support was unhelpful.',
                '',
                None,
                'Billing dispute regarding unauthorized transaction.',
                'Loan application was rejected without proper explanation.',
                'High interest rates on personal loan.',
                'Issues with loan repayment schedule.',
                'Mortgage processing delay.',
                'Problems with mortgage refinancing.'
            ],
            'Company': ['Bank A', 'Bank B', 'Bank A', 'Bank C', 'Bank B'] * 2,
            'State': ['CA', 'NY', 'TX', 'FL', 'IL'] * 2,
            'Issue': ['Billing dispute', 'Service issue', 'Billing', 'Other', 'Fraud'] * 2
        })
    
    @pytest.fixture
    def preprocessor(self, sample_data, tmp_path):
        """Create DataPreprocessor instance with sample data"""
        data_path = tmp_path / "test_complaints.csv"
        sample_data.to_csv(data_path, index=False)
        return DataPreprocessor(str(data_path))
    
    def test_init_with_valid_path(self, preprocessor):
        """Test initialization with valid file path"""
        assert preprocessor.data_path.exists()
        assert preprocessor.df is None
        assert preprocessor.processed_df is None
    
    def test_init_with_invalid_path(self):
        """Test initialization with invalid file path raises error"""
        with pytest.raises(FileNotFoundError):
            DataPreprocessor("/nonexistent/path/complaints.csv")
    
    def test_load_data_success(self, preprocessor):
        """Test successful data loading"""
        df = preprocessor.load_data()
        
        assert df is not None
        assert len(df) == 10
        assert 'complaint_id' in df.columns
        assert 'Consumer complaint narrative' in df.columns
        assert preprocessor.df is not None
    
    @patch('pandas.read_csv')
    def test_load_data_failure(self, mock_read_csv, preprocessor):
        """Test data loading failure"""
        mock_read_csv.side_effect = Exception("File read error")
        
        with pytest.raises(Exception):
            preprocessor.load_data()
    
    def test_clean_text_standard_case(self, preprocessor):
        """Test text cleaning with standard input"""
        input_text = "I AM WRITING TO FILE A COMPLAINT about BILLING. Contact: test@email.com"
        expected = "about billing contact"
        
        result = preprocessor.clean_text(input_text)
        
        assert result == expected
        assert result.islower()
        assert "@" not in result
        assert "writing to file a complaint" not in result
    
    def test_clean_text_with_none(self, preprocessor):
        """Test text cleaning with None input"""
        result = preprocessor.clean_text(None)
        assert result == ""
    
    def test_clean_text_with_empty_string(self, preprocessor):
        """Test text cleaning with empty string"""
        result = preprocessor.clean_text("")
        assert result == ""
    
    def test_clean_text_with_special_characters(self, preprocessor):
        """Test text cleaning removes special characters"""
        input_text = "Special #$%^&*() characters @replaced"
        result = preprocessor.clean_text(input_text)
        
        # Should only contain alphanumeric and basic punctuation
        assert all(c.isalnum() or c.isspace() or c in '.,!?\'"- ' for c in result)
    
    def test_clean_text_phone_numbers(self, preprocessor):
        """Test text cleaning removes phone numbers"""
        input_text = "Call me at 123-456-7890 or (987) 654-3210"
        result = preprocessor.clean_text(input_text)
        
        assert "123-456-7890" not in result
        assert "(987) 654-3210" not in result
    
    def test_filter_by_products_success(self, preprocessor):
        """Test filtering by specific products"""
        preprocessor.load_data()
        original_count = len(preprocessor.df)
        
        products_to_keep = ['Credit card', 'Personal Loan']
        preprocessor.filter_by_products(products_to_keep)
        
        assert len(preprocessor.df) < original_count
        assert all(product in products_to_keep for product in preprocessor.df['Product'].unique())
    
    def test_filter_by_products_no_match(self, preprocessor):
        """Test filtering when no products match"""
        preprocessor.load_data()
        
        products_to_keep = ['Nonexistent Product']
        preprocessor.filter_by_products(products_to_keep)
        
        assert len(preprocessor.df) == 0
    
    def test_remove_empty_narratives(self, preprocessor):
        """Test removal of empty narratives"""
        preprocessor.load_data()
        original_count = len(preprocessor.df)
        
        preprocessor.remove_empty_narratives()
        
        assert len(preprocessor.df) < original_count
        assert preprocessor.df['has_narrative'].all()
        assert preprocessor.df['Consumer complaint narrative'].notna().all()
        assert (preprocessor.df['Consumer complaint narrative'].str.strip() != '').all()
    
    def test_standardize_product_categories(self, preprocessor):
        """Test product category standardization"""
        preprocessor.load_data()
        
        preprocessor.standardize_product_categories()
        
        assert 'product_category' in preprocessor.df.columns
        assert preprocessor.df['product_category'].notna().all()
        
        # Check mapping
        credit_card_rows = preprocessor.df[preprocessor.df['Product'] == 'Credit card']
        if not credit_card_rows.empty:
            assert (credit_card_rows['product_category'] == 'Credit Card').all()
    
    def test_apply_text_cleaning(self, preprocessor):
        """Test application of text cleaning to all narratives"""
        preprocessor.load_data()
        preprocessor.remove_empty_narratives()
        
        original_narratives = preprocessor.df['Consumer complaint narrative'].copy()
        preprocessor.apply_text_cleaning()
        
        assert 'cleaned_narrative' in preprocessor.df.columns
        assert preprocessor.df['cleaned_narrative'].notna().all()
        
        # Check that cleaning was applied
        for orig, cleaned in zip(original_narratives, preprocessor.df['cleaned_narrative']):
            if pd.notna(orig):
                assert cleaned.islower()
                assert len(cleaned) <= len(str(orig))
    
    def test_add_text_statistics(self, preprocessor):
        """Test addition of text statistics columns"""
        preprocessor.load_data()
        preprocessor.remove_empty_narratives()
        preprocessor.apply_text_cleaning()
        
        preprocessor.add_text_statistics()
        
        assert 'word_count' in preprocessor.df.columns
        assert 'char_count' in preprocessor.df.columns
        
        assert preprocessor.df['word_count'].dtype == np.int64
        assert preprocessor.df['char_count'].dtype == np.int64
        
        # Word count should be less than or equal to character count
        assert (preprocessor.df['word_count'] <= preprocessor.df['char_count']).all()
    
    def test_preprocess_complete_pipeline(self, preprocessor):
        """Test complete preprocessing pipeline"""
        processed_df = preprocessor.preprocess()
        
        assert processed_df is not None
        assert preprocessor.processed_df is not None
        
        # Check required columns exist
        required_columns = ['cleaned_narrative', 'product_category', 'word_count', 'char_count']
        for col in required_columns:
            assert col in processed_df.columns
        
        # Check data quality
        assert processed_df['cleaned_narrative'].notna().all()
        assert (processed_df['cleaned_narrative'].str.len() > 0).all()
        assert processed_df['product_category'].notna().all()
        assert (processed_df['word_count'] > 0).all()
    
    def test_save_processed_data_success(self, preprocessor, tmp_path):
        """Test successful save of processed data"""
        preprocessor.preprocess()
        
        output_path = tmp_path / "processed_complaints.csv"
        preprocessor.save_processed_data(str(output_path))
        
        assert output_path.exists()
        
        # Load and verify saved data
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == len(preprocessor.processed_df)
        assert 'cleaned_narrative' in saved_df.columns
    
    def test_save_processed_data_no_data(self, preprocessor, tmp_path):
        """Test save fails when no processed data exists"""
        output_path = tmp_path / "processed_complaints.csv"
        
        with pytest.raises(ValueError):
            preprocessor.save_processed_data(str(output_path))
    
    def test_data_validation_error_handling(self, preprocessor):
        """Test error handling with invalid data"""
        # Create invalid data
        invalid_df = pd.DataFrame({
            'Product': [None] * 5,
            'Consumer complaint narrative': [None] * 5
        })
        
        with patch.object(preprocessor, 'df', invalid_df):
            with pytest.raises(Exception):
                preprocessor.preprocess()
    
    @patch('pandas.DataFrame.to_csv')
    def test_save_processed_data_io_error(self, mock_to_csv, preprocessor, tmp_path):
        """Test IO error during save"""
        preprocessor.preprocess()
        
        mock_to_csv.side_effect = IOError("Disk full")
        output_path = tmp_path / "processed_complaints.csv"
        
        with pytest.raises(IOError):
            preprocessor.save_processed_data(str(output_path))
    
    def test_memory_efficiency(self, preprocessor):
        """Test that preprocessing doesn't create excessive copies"""
        preprocessor.load_data()
        original_memory = preprocessor.df.memory_usage(deep=True).sum()
        
        processed_df = preprocessor.preprocess()
        processed_memory = processed_df.memory_usage(deep=True).sum()
        
        # Processed data should not be significantly larger
        # Allow for some increase due to new columns
        assert processed_memory <= original_memory * 1.5
    
    def test_reproducibility(self, preprocessor):
        """Test that preprocessing is reproducible"""
        # Run preprocessing twice
        result1 = preprocessor.preprocess()
        
        # Reset and run again
        preprocessor2 = DataPreprocessor(str(preprocessor.data_path))
        result2 = preprocessor2.preprocess()
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_edge_cases(self):
        """Test edge cases in text cleaning"""
        preprocessor = DataPreprocessor("dummy_path")
        
        test_cases = [
            ("   Multiple   spaces   ", "multiple spaces"),
            ("LINE1\nLINE2\nLINE3", "line1 line2 line3"),
            ("Special@Chars#Here$", "specialcharshere"),
            ("123 Main St., Apt 4B", "main st apt b"),
            ("", ""),
            ("   ", ""),
            (None, "")
        ]
        
        for input_text, expected in test_cases:
            result = preprocessor.clean_text(input_text)
            assert result == expected, f"Failed for input: {input_text}"
    
    def test_performance_large_dataset(self, tmp_path):
        """Test performance with larger dataset"""
        # Create larger test dataset
        large_data = pd.DataFrame({
            'complaint_id': range(1000),
            'Product': ['Credit card'] * 1000,
            'Consumer complaint narrative': ['Test complaint ' + str(i) for i in range(1000)],
            'Company': ['Test Bank'] * 1000
        })
        
        data_path = tmp_path / "large_complaints.csv"
        large_data.to_csv(data_path, index=False)
        
        preprocessor = DataPreprocessor(str(data_path))
        
        import time
        start_time = time.time()
        processed_df = preprocessor.preprocess()
        end_time = time.time()
        
        assert len(processed_df) == 1000
        assert (end_time - start_time) < 10  # Should process 1000 records in under 10 seconds
    
    def test_unicode_handling(self, preprocessor):
        """Test handling of unicode characters"""
        test_text = "Complaint with emoji 😊 and foreign characters: café résumé naïve"
        result = preprocessor.clean_text(test_text)
        
        # Should preserve meaningful unicode characters
        assert "café" in result.lower()
        assert "😊" not in result  # Emojis should be removed
    
    def test_concurrent_preprocessing(self, preprocessor):
        """Test that preprocessing can be called multiple times"""
        # First call
        result1 = preprocessor.preprocess()
        assert result1 is not None
        
        # Reset and call again
        preprocessor.df = None
        preprocessor.processed_df = None
        
        result2 = preprocessor.preprocess()
        assert result2 is not None
        
        # Results should be equivalent
        pd.testing.assert_frame_equal(result1, result2)