import React, { useEffect, useState } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
        </div>
      </div>
    </header>
  );
}

interface Submission {
  user: string;
  time: string;
  is_fastest: boolean;
}

interface Problem {
  name: string;
  deadline: string;
  submissions: Submission[];
}

interface GPUType {
  name: string;
  problems: Problem[];
}

function LeaderboardSection() {
  const [gpuTypes, setGPUTypes] = useState<GPUType[]>([]);
  const [timestamp, setTimestamp] = useState<string>('');

  useEffect(() => {
    console.log('Fetching leaderboard data...');
    Promise.any([
      fetch('/discord-cluster-manager/leaderboard/table.html'),
      fetch('/leaderboard/table.html')
    ])
      .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.text();
      })
      .then(html => {
        if (!html) {
          throw new Error('Empty response');
        }
        
        console.log('Received HTML:', html.substring(0, 200) + '...');
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        
        const timestampEl = doc.querySelector('.timestamp');
        if (timestampEl) {
          setTimestamp(timestampEl.textContent?.replace('Last updated: ', '') || '');
        }

        // Parse GPU types and their problems
        const gpuTypeElements = doc.querySelectorAll('.gpu-type');
        const parsedGPUTypes = Array.from(gpuTypeElements).map(gpuTypeEl => {
          const name = gpuTypeEl.getAttribute('data-gpu-type') || '';
          const problemElements = gpuTypeEl.querySelectorAll('.problem');
          
          const problems = Array.from(problemElements).map(problemEl => {
            const problemName = problemEl.getAttribute('data-problem-name') || '';
            const deadline = problemEl.querySelector('.problem-deadline')?.textContent?.replace('Deadline: ', '') || '';
            const submissionElements = problemEl.querySelectorAll('.submission');
            
            const submissions = Array.from(submissionElements).map(subEl => ({
              user: subEl.getAttribute('data-user') || '',
              time: subEl.getAttribute('data-time') || '',
              is_fastest: subEl.classList.contains('fastest')
            }));
            
            return {
              name: problemName,
              deadline,
              submissions
            };
          });
          
          return {
            name,
            problems
          };
        });

        console.log('Setting GPU types:', parsedGPUTypes);
        setGPUTypes(parsedGPUTypes);
      })
      .catch(error => {
        console.error('Error fetching leaderboard:', error);
      });
  }, []);

  return (
    <section className={styles.leaderboardSection}>
      <div className="container">
        <h2>Active Leaderboards</h2>
        {timestamp && (
          <div className={styles.timestamp}>Last updated: {timestamp}</div>
        )}
        
        <div className={styles.gpuTypesContainer}>
          {gpuTypes.map((gpuType) => (
            <div 
              key={gpuType.name}
              className={styles.gpuTypeGroup}
            >
              <h2 className={styles.gpuTypeTitle}>{gpuType.name}</h2>
              <div className={styles.problemsContainer}>
                {gpuType.problems.map((problem) => (
                  <div key={problem.name} className={styles.problemGroup}>
                    <h3 className={styles.problemName}>{problem.name}</h3>
                    <div className={styles.problemDeadline}>Deadline: {problem.deadline}</div>
                    <div className={styles.submissionsList}>
                      {problem.submissions.map((submission, idx) => (
                        <div 
                          key={`${submission.user}-${idx}`}
                          className={clsx(styles.submission, {
                            [styles.fastest]: submission.is_fastest
                          })}
                        >
                          {submission.user} - {submission.time}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}>
      <HomepageHeader />
      <main>
        <LeaderboardSection />
      </main>
    </Layout>
  );
}