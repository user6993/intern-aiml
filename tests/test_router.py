import unittest

from router import IntakeRouter, ROUTES, score_completeness


class IntakeRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = IntakeRouter.from_csv()

    def test_routes_clear_examples_to_expected_service(self):
        examples = {
            "We want an AI agent to summarize support tickets and draft answers for approval.": "ai_automation",
            "Build a multi-tenant SaaS web application with user accounts and a backend API.": "software_platform",
            "Migrate our virtual machines to cloud infrastructure and set up CI/CD monitoring.": "cloud_devops",
            "Connect CRM records and campaign analytics to improve our marketing funnel.": "data_marketing",
            "Create technical courses and recommend coding exercises from student quiz progress.": "learning_tech",
        }
        for brief, expected in examples.items():
            with self.subTest(expected=expected):
                self.assertEqual(self.router.predict(brief).label, expected)

    def test_scores_are_ranked_and_sum_to_one(self):
        result = self.router.predict(
            "Build an AI agent that automates repetitive support email and escalates exceptions."
        )
        self.assertEqual(len(result.scores), len(ROUTES))
        self.assertEqual(
            list(result.scores),
            sorted(result.scores, key=lambda item: item[1], reverse=True),
        )
        self.assertAlmostEqual(sum(score for _, score in result.scores), 1.0, places=6)

    def test_empty_or_short_brief_is_rejected(self):
        for brief in ("", "too short"):
            with self.subTest(brief=brief):
                with self.assertRaises(ValueError):
                    self.router.predict(brief)

    def test_oversized_brief_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "5000 characters or fewer"):
            self.router.predict("A" * 5001)

    def test_explanation_terms_come_from_the_brief(self):
        brief = "Build an AI agent for support ticket automation."
        result = self.router.predict(brief)
        self.assertEqual(result.label, "ai_automation")
        self.assertTrue(result.evidence)
        self.assertTrue(all(term in brief.lower() for term in result.evidence))

    def test_completeness_score_and_missing_fields(self):
        score, missing = score_completeness(
            {
                "project_brief": "Build a customer support automation workflow.",
                "desired_outcome": "Reduce handling time.",
                "current_stack": "",
                "timeline": "This quarter",
                "success_metric": "",
            }
        )
        self.assertEqual(score, 70)
        self.assertEqual(missing, ["Current tools or systems", "Success measure"])

    def test_complete_intake_scores_one_hundred(self):
        score, missing = score_completeness(
            {
                "project_brief": "Build a customer support automation workflow.",
                "desired_outcome": "Reduce handling time.",
                "current_stack": "A help desk",
                "timeline": "This quarter",
                "success_metric": "Average resolution time",
            }
        )
        self.assertEqual(score, 100)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
