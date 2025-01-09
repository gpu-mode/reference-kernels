import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>

        <center>
        <div align="center" className="header-links">
          <br></br>
          <a href="https://github.com/gpu-mode/discord-cluster-manager/graphs/contributors"><img src="https://img.shields.io/github/contributors/gpu-mode/discord-cluster-manager?style=for-the-badge&color=blue" alt="Contributors" /></a>
          <a href="https://github.com/gpu-mode/discord-cluster-manager/stargazers"><img src="https://img.shields.io/github/stars/gpu-mode/discord-cluster-manager?style=for-the-badge&color=blue" alt="Stargazers" /></a>
          <a href="https://github.com/gpu-mode/discord-cluster-manager/blob/main/LICENSE"><img src="https://img.shields.io/github/license/gpu-mode/discord-cluster-manager?style=for-the-badge&color=blue" alt="MIT License" /></a>
          <br/>
          <a href="https://discord.gg/"><img src="https://img.shields.io/badge/Discord-Join%20Us-purple?logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord community" /></a>
          <br/>
          <a href="docs/intro"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Getting started." /></a>
          <br/>
        </div>
        </center>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
