from floppy.flop_analysis.core.analyzer import FlopAnalyzer


def main():
    analyzer = FlopAnalyzer()

    try:
        # Analyze a directory of Python files
        results = analyzer.analyze_directory("~/tbp/tbp.monty/src")

        # Save results to CSV
        output_file = analyzer.save_results(results, "./analysis_results")


        print(f"\nDetailed results saved to: {output_file}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()
