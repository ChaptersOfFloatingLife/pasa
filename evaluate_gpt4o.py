#!/usr/bin/env python3
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import argparse
from datetime import datetime, timedelta
from paper_agent import PaperAgent
from models import GPT4Agent
from utils import keep_letters, cal_micro

def evaluate_gpt4o(args):
    """Run evaluation of PaSa-GPT-4o baseline."""
    # Initialize GPT-4 agents
    crawler = GPT4Agent()
    selector = GPT4Agent()
    
    # Track metrics
    crawler_recalls = []
    precisions = []
    recalls = []
    recalls_100 = []
    recalls_50 = []
    recalls_20 = []
    actions = []
    scores = []
    
    # Process test data
    with open(args.test_data) as f:
        papers = f.readlines()
        if args.paper_limit:
            papers = papers[:args.paper_limit]  # Limit papers if specified
        for idx, line in enumerate(papers):
            data = json.loads(line)
            print(data)
            print()
            end_date = data['source_meta']['published_time']
            end_date = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=7)
            end_date = end_date.strftime("%Y%m%d")
            
            # Run paper agent
            paper_agent = PaperAgent(
                user_query=data['question'],
                crawler=crawler,
                selector=selector,
                end_date=end_date,
                expand_layers=args.expand_layers,
                search_queries=args.search_queries,
                search_papers=args.search_papers,
                expand_papers=args.expand_papers,
                threads_num=args.threads_num
            )
            if "answer" in data:
                paper_agent.root.extra["answer"] = data["answer"]
            
            paper_agent.run()
            
            # Save results
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, f"{idx}.json"), "w") as f:
                    json.dump(
                        paper_agent.root.todic(),
                        f,
                        indent=2
                    )
            
            # Calculate metrics
            paper_root = paper_agent.root.todic()
            crawled_papers = []
            crawled_paper_set = set()
            selected_paper_set = set()
            queue = [paper_root]
            action = 0
            score = []
            answer_paper_set = set([keep_letters(paper) for paper in paper_root["extra"]["answer"]])
            
            while len(queue) > 0:
                node, queue = queue[0], queue[1:]
                action += len(node["child"])
                total_score = 0
                for _, v in node["child"].items():
                    total_score -= 0.1
                    for i in v:
                        queue.append(i)
                        if i["select_score"] > 0.5:
                            selected_paper_set.add(keep_letters(i["title"]))
                            total_score += 1
                        if keep_letters(i["title"]) not in crawled_paper_set:
                            crawled_paper_set.add(keep_letters(i["title"]))
                            crawled_papers.append([keep_letters(i["title"]), i["select_score"]])
                score.append(total_score)
            
            actions.append(action)
            scores.append(sum(score) / len(score) if len(score) > 0 else 0)
            
            # Calculate recall metrics
            crawled_papers.sort(key=lambda x: x[1], reverse=True)
            crawled_20, crawled_50, crawled_100 = set(), set(), set()
            for i in range(100):
                if i >= len(crawled_papers):
                    break
                if i < 20:
                    crawled_20.add(crawled_papers[i][0])
                if i < 50:
                    crawled_50.add(crawled_papers[i][0])
                crawled_100.add(crawled_papers[i][0])

            crawled_res = cal_micro(crawled_paper_set, answer_paper_set)
            selected_res = cal_micro(selected_paper_set, answer_paper_set)
            crawled_20_res = cal_micro(crawled_20, answer_paper_set)
            crawled_50_res = cal_micro(crawled_50, answer_paper_set)
            crawled_100_res = cal_micro(crawled_100, answer_paper_set)

            crawler_recalls.append(crawled_res[0] / (crawled_res[0] + crawled_res[2] if (crawled_res[0] + crawled_res[2]) > 0 else 1e-9))
            precisions.append(selected_res[0] / (selected_res[0] + selected_res[1] if (selected_res[0] + selected_res[1]) > 0 else 1e-9))
            recalls.append(selected_res[0] / (selected_res[0] + selected_res[2] if (selected_res[0] + selected_res[2]) > 0 else 1e-9))
            recalls_100.append(crawled_100_res[0] / (crawled_100_res[0] + crawled_100_res[2] if (crawled_100_res[0] + crawled_100_res[2]) > 0 else 1e-9))
            recalls_50.append(crawled_50_res[0] / (crawled_50_res[0] + crawled_50_res[2] if (crawled_50_res[0] + crawled_50_res[2]) > 0 else 1e-9))
            recalls_20.append(crawled_20_res[0] / (crawled_20_res[0] + crawled_20_res[2] if (crawled_20_res[0] + crawled_20_res[2]) > 0 else 1e-9))
    
    # Print final metrics
    print("=== PaSa-GPT-4o Evaluation Results ===")
    print("Format: crawler_recall & precision & recall & recall@100 & recall@50 & recall@20")
    print("{} & {} & {} & {} & {} & {}".format(
        round(sum(crawler_recalls) / len(crawler_recalls), 4),
        round(sum(precisions) / len(precisions), 4),
        round(sum(recalls) / len(recalls), 4),
        round(sum(recalls_100) / len(recalls_100), 4),
        round(sum(recalls_50) / len(recalls_50), 4),
        round(sum(recalls_20) / len(recalls_20), 4),
    ))
    print("\nFormat: crawler_recall & actions & scores & precision & recall")
    print("{} & {} & {} & {} & {}".format(
        round(sum(crawler_recalls) / len(crawler_recalls), 4),
        round(sum(actions) / len(actions), 4),
        round(sum(scores) / len(scores), 4),
        round(sum(precisions) / len(precisions), 4),
        round(sum(recalls) / len(recalls), 4),
    ))

def main():
    parser = argparse.ArgumentParser(description='Evaluate PaSa-GPT-4o baseline')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data file')
    parser.add_argument('--output_dir', type=str, default='results_gpt4o',
                       help='Directory to save evaluation results')
    parser.add_argument('--expand_layers', type=int, default=2,
                       help='Number of expansion layers')
    parser.add_argument('--search_queries', type=int, default=5,
                       help='Number of search queries')
    parser.add_argument('--search_papers', type=int, default=10,
                       help='Number of papers per query')
    parser.add_argument('--expand_papers', type=int, default=20,
                       help='Number of papers per layer')
    parser.add_argument('--threads_num', type=int, default=20,
                       help='Number of threads for parallel processing')
    parser.add_argument('--paper_limit', type=int, default=None, 
                       help='Limit the number of papers to process (for testing)')
    
    args = parser.parse_args()
    evaluate_gpt4o(args)

if __name__ == "__main__":
    main()
