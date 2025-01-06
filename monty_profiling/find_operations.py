from floppy.flop_analysis.core.analyzer import FlopAnalyzer


def main():
    analyzer = FlopAnalyzer()

    try:
        # Analyze a directory of Python files
        results = analyzer.analyze_directory("/Users/hlee/tbp/tbp.monty/src")

        # Save results to CSV
        output_file = analyzer.save_results(results, "./analysis_results")

        # Print summary
        print("\nAnalysis Summary:")
        print(f"Total files analyzed: {results['summary']['total_files_analyzed']}")
        print(f"Total FLOP operations: {results['summary']['total_flop_operations']}")

        print("\nMost common FLOP operations:")
        for op in results["summary"]["most_common_operations"]:
            print(f"- {op['operation']}: {op['count']} occurrences")

        print("\nFiles with most FLOP operations:")
        for file_info in results["summary"]["files_with_most_flops"]:
            print(f"- {file_info['file']}: {file_info['flop_count']} operations")

        print(f"\nDetailed results saved to: {output_file}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()
