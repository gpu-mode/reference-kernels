import React, { useEffect, useState } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
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
  rank: number | null;
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
  const [currentPages, setCurrentPages] = useState<Record<string, number>>({});
  const ITEMS_PER_PAGE = 10;

  const handlePageChange = (problemKey: string, delta: number) => {
    setCurrentPages(prev => ({
      ...prev,
      [problemKey]: Math.max(0, (prev[problemKey] || 0) + delta)
    }));
  };

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
              rank: parseInt(subEl.getAttribute('data-rank') || '-1')
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

  const calculateTopPerformers = () => {
    const medalCounts: Record<string, { gold: number; silver: number; bronze: number; points: number }> = {};
    
    gpuTypes.forEach(gpuType => {
      gpuType.problems.forEach(problem => {
        problem.submissions.forEach(submission => {
          if (!medalCounts[submission.user]) {
            medalCounts[submission.user] = { gold: 0, silver: 0, bronze: 0, points: 0 };
          }
          if (submission.rank === 1) {
            medalCounts[submission.user].gold++;
            medalCounts[submission.user].points += 3;
          }
          if (submission.rank === 2) {
            medalCounts[submission.user].silver++;
            medalCounts[submission.user].points += 2;
          }
          if (submission.rank === 3) {
            medalCounts[submission.user].bronze++;
            medalCounts[submission.user].points += 1;
          }
        });
      });
    });

    return Object.entries(medalCounts)
      .sort(([, a], [, b]) => b.points - a.points)
      .slice(0, 3)
      .map(([user, medals]) => [user, { gold: medals.gold, silver: medals.silver, bronze: medals.bronze }]);
  };

  return (
    <section className={styles.leaderboardSection}>
      <div className="container">
        <h2 style={{
          fontSize: '4.2rem',
          textAlign: 'center',
          fontWeight: '800',
          marginBottom: '0.5rem',
          color: 'black'
        }}>GPU_MODE Kernel Leaderboards</h2>
        {timestamp && (
          <div className={styles.timestamp}>Last updated: {timestamp}</div>
        )}

        <div style={{
          background: 'var(--ifm-color-emphasis-100)',
          borderRadius: '8px',
          padding: '2rem',
          marginBottom: '2rem',
          textAlign: 'center'
        }}>
          <h3 style={{ 
            marginBottom: '1.5rem', 
            color: 'var(--ifm-color-primary)',
            fontSize: '2.5rem',
            fontWeight: '600'
          }}>
            Top Active Participants
          </h3>
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'flex-end',
            gap: '1rem',
            minHeight: '250px',
            padding: '2rem 1rem'
          }}>
            {calculateTopPerformers().map(([user, medals], index) => {
              const podiumHeight = index === 0 ? '120px' : index === 1 ? '90px' : '60px';
              const order = index === 0 ? 2 : index === 1 ? 1 : 3;
              
              return (
                <div key={user} style={{
                  display: 'flex',
                  flexDirection: 'column',
                  order,
                  transition: 'all 0.3s ease'
                }}>
                  <div style={{
                    background: 'white',
                    padding: '1.5rem',
                    borderRadius: '8px 8px 0 0',
                    minWidth: '200px',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                    zIndex: 1
                  }}>
                    <div style={{ 
                      fontSize: '1.2rem', 
                      fontWeight: 'bold', 
                      marginBottom: '0.5rem',
                      position: 'relative' 
                    }}>
                      {index === 0 && (
                        <span style={{
                          position: 'absolute',
                          top: '-1.2rem',
                          left: '50%',
                          transform: 'translateX(-50%)'
                        }}>
                          👑
                        </span>
                      )}
                      <div style={{ 
                        display: 'inline-block',
                        whiteSpace: 'nowrap'
                      }}>
                        {user.split('').map((char, i) => (
                          <span
                            key={i}
                            style={{
                              display: 'inline-block',
                              animation: `waveText 1s ease-in-out infinite`,
                              animationDelay: `${i * 0.1}s`
                            }}
                          >
                            {char}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
                      <span>🥇 {medals.gold}</span>
                      <span>🥈 {medals.silver}</span>
                      <span>🥉 {medals.bronze}</span>
                    </div>
                  </div>
                  <div style={{
                    background: index === 0 ? '#FFD700' : index === 1 ? '#C0C0C0' : '#CD7F32',
                    height: podiumHeight,
                    width: '200px',
                    borderRadius: '0 0 8px 8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '2rem',
                    fontWeight: 'bold',
                    color: 'white',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                  }}>
                    {index === 0 ? '1st' : index === 1 ? '2nd' : '3rd'}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className={styles.gpuTypesContainer}>
          {gpuTypes.map((gpuType) => (
            <div
              key={gpuType.name}
              className={styles.gpuTypeGroup}
            >
              <h2 className={styles.gpuTypeTitle}>{gpuType.name}</h2>
              <div className={styles.problemsContainer}>
                {gpuType.problems.map((problem) => {
                  const problemKey = `${gpuType.name}-${problem.name}`;
                  const currentPage = currentPages[problemKey] || 0;
                  const startIdx = currentPage * ITEMS_PER_PAGE;
                  const visibleSubmissions = problem.submissions.slice(
                    startIdx,
                    startIdx + ITEMS_PER_PAGE
                  );
                  const hasNextPage = startIdx + ITEMS_PER_PAGE < problem.submissions.length;
                  const hasPrevPage = currentPage > 0;

                  return (
                    <div key={problem.name} className={styles.problemGroup}>
                      <h3 className={styles.problemName}>{problem.name}</h3>
                      <div className={styles.problemDeadline}>Deadline: {problem.deadline}</div>
                      <div className={styles.submissionsList}>
                        {visibleSubmissions.map((submission, idx) => (
                          <div
                            key={`${submission.user}-${idx}`}
                            className={clsx(styles.submission, {
                              [styles.first]: submission.rank === 1,
                              [styles.second]: submission.rank === 2,
                              [styles.third]: submission.rank === 3
                            })}
                          >
                            <span style={{ marginRight: '1rem', minWidth: '2rem' }}>
                              {submission.rank === 1 && '🥇'}
                              {submission.rank === 2 && '🥈'}
                              {submission.rank === 3 && '🥉'}
                              {submission.rank > 3 && `${submission.rank}.`}
                            </span>
                            <span style={{
                              flex: 1,
                              fontWeight: submission.rank <= 5 ? '700' : 'normal', 
                              fontSize: '1.2rem'
                            }}>
                              {submission.user}
                            </span>
                            <span style={{
                              color: submission.rank === 1 ? '#FFB800' : // Gold with red glow
                                submission.rank === 2 ? '#C0C0C0' : // Silver
                                  submission.rank === 3 ? '#CD7F32' : // Bronze
                                    submission.rank <= 5 ? '#000000' : // Bold black for 4-5
                                      'var(--ifm-color-emphasis-600)', // Gray for others
                              fontSize: '1.2rem',
                              fontWeight: submission.rank <= 5 ? '700' : '500',
                              fontFamily: 'monospace',
                              minWidth: '8ch',
                              textAlign: 'right'
                            }}>
                              {submission.time}
                            </span>
                          </div>
                        ))}
                      </div>
                      <div className={styles.paginationControls}>
                        <button
                          onClick={() => handlePageChange(problemKey, -1)}
                          disabled={!hasPrevPage}
                          className={styles.pageButton}
                        >
                          ← Previous
                        </button>
                        <span className={styles.pageInfo}>
                          Top {startIdx + 1}-{Math.min(startIdx + ITEMS_PER_PAGE, problem.submissions.length)}
                        </span>
                        <button
                          onClick={() => handlePageChange(problemKey, 1)}
                          disabled={!hasNextPage}
                          className={styles.pageButton}
                        >
                          Next →
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}>
      <main>
        <LeaderboardSection />
      </main>
    </Layout>
  );
}

<style>
  {`
    @keyframes waveText {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-3px); }
    }
  `}
</style>
