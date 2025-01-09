import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import { Demo } from "../Demo/Demo";
import styles from './styles.module.css';
import "../../css/homepageHeader.css";

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Introduction Video',
    Svg: require('@site/static/img/logo.svg').default,
    description: (
      <>
        Put a quick video introduction here. Then point to <code>docs</code> directory.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          <div className={clsx('col col')}>
            <div className="text--center padding-horiz--md">
              <p>{<Demo />}</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
