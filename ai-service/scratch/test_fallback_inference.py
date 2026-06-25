import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.clarification_planner import (
    classify_action_class,
    infer_behavior_dimensions,
    _purpose_specific_fallback,
)

class TestFallbackInference(unittest.TestCase):

    def test_action_class_classification(self):
        # Evaluation cases
        self.assertEqual(classify_action_class("I want an app that reviews scholarship essays"), "Evaluation")
        self.assertEqual(classify_action_class("audit restaurant menus"), "Evaluation")
        
        # Analysis cases
        self.assertEqual(classify_action_class("analyze pitch decks"), "Analysis")
        
        # Recommendation cases
        self.assertEqual(classify_action_class("recommend Airbnb pricing"), "Recommendation")
        
        # Transformation cases
        self.assertEqual(classify_action_class("Convert books into audiobooks"), "Transformation")

    def test_inferred_dimensions_evaluation(self):
        purpose = "I want an app that reviews scholarship essays"
        dims = infer_behavior_dimensions(purpose, "text")
        
        fields = [d["field"] for d in dims]
        self.assertIn("feedback_focus", fields)
        self.assertIn("evaluation_standard", fields)
        self.assertNotIn("writing_style", fields)
        self.assertNotIn("writing_tone", fields)
        
        # Verify first fallback question is not generic and is feedback_focus
        fallback_plan = _purpose_specific_fallback(purpose, {}, "text")
        self.assertEqual(fallback_plan["selected_key"], "feedback_focus")
        self.assertIn("criteria or aspects should the review focus on", fallback_plan["selected_question"])

    def test_inferred_dimensions_audiobook(self):
        purpose = "Convert books into audiobooks"
        dims = infer_behavior_dimensions(purpose, "audio")
        
        fields = [d["field"] for d in dims]
        self.assertIn("narration_style", fields)
        self.assertIn("voice_type", fields)
        
        # Check audiobook sub-case question
        fallback_plan = _purpose_specific_fallback(purpose, {}, "audio")
        self.assertEqual(fallback_plan["selected_key"], "narration_style")
        self.assertIn("narration style should the audiobook use", fallback_plan["selected_question"])

    def test_inferred_dimensions_recommendation(self):
        purpose = "recommend Airbnb pricing"
        dims = infer_behavior_dimensions(purpose, "text")
        
        fields = [d["field"] for d in dims]
        self.assertIn("recommendation_criteria", fields)
        self.assertIn("output_limit", fields)
        
        fallback_plan = _purpose_specific_fallback(purpose, {}, "text")
        self.assertEqual(fallback_plan["selected_key"], "recommendation_criteria")
        self.assertIn("preferences or constraints should guide the recommendations", fallback_plan["selected_question"])

    def test_modality_fallback_emergency(self):
        # A purpose that matches no Action Class
        purpose = "an app for reading"
        dims = infer_behavior_dimensions(purpose, "image")
        
        fields = [d["field"] for d in dims]
        self.assertEqual(fields, ["visual_style"])
        
        fallback_plan = _purpose_specific_fallback(purpose, {}, "image")
        self.assertEqual(fallback_plan["selected_key"], "visual_style")
        self.assertIn("visual style or aesthetic theme", fallback_plan["selected_question"])

if __name__ == "__main__":
    unittest.main()
