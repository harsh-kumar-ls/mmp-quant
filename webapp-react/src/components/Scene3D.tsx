/* The hero scene (10 Nov 2026): a clay TV whose screen shows a slide, a speech
   bubble saying the same line (that is redundancy), and the red pen that marks
   it. Built from primitives — no model files, no network. Lazy-loaded, so only
   the home page pays for three.js. */
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, Float, PresentationControls, RoundedBox } from "@react-three/drei";
import { useEffect, useRef } from "react";
import type { Group } from "three";

const CLAY = { body: "#e7d6bf", dark: "#2c3b33", line: "#f1ead9", hl: "#ffe36e", pen: "#ad3119", ink: "#1d1a16", white: "#fbf8f2", leg: "#b9a68b" };

function Line({ x = 0, y, w, color = CLAY.line, z = 0.071 }: { x?: number; y: number; w: number; color?: string; z?: number }) {
  return <RoundedBox args={[w, 0.075, 0.02]} radius={0.02} position={[x, y, z]}><meshStandardMaterial color={color} roughness={0.9} /></RoundedBox>;
}

function TV() {
  return (
    <group>
      <RoundedBox args={[2.3, 1.62, 0.9]} radius={0.22} smoothness={6} castShadow>
        <meshStandardMaterial color={CLAY.body} roughness={0.85} />
      </RoundedBox>
      {/* screen: a slide with three lines, the middle one highlighted */}
      <group position={[-0.18, 0.02, 0.42]}>
        <RoundedBox args={[1.6, 1.12, 0.12]} radius={0.08}><meshStandardMaterial color={CLAY.dark} roughness={0.6} /></RoundedBox>
        <Line x={-0.22} y={0.3} w={0.9} />
        <RoundedBox args={[1.12, 0.16, 0.02]} radius={0.04} position={[-0.03, 0.03, 0.07]}><meshStandardMaterial color={CLAY.hl} roughness={0.8} /></RoundedBox>
        <Line x={-0.06} y={0.03} w={0.96} color={CLAY.ink} z={0.085} />
        <Line x={-0.3} y={-0.24} w={0.74} />
      </group>
      {/* knobs */}
      <mesh position={[0.86, 0.32, 0.46]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.11, 0.11, 0.08, 32]} /><meshStandardMaterial color={CLAY.pen} roughness={0.7} /></mesh>
      <mesh position={[0.86, 0.0, 0.46]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.09, 0.09, 0.08, 32]} /><meshStandardMaterial color={CLAY.leg} roughness={0.8} /></mesh>
      <RoundedBox args={[0.26, 0.07, 0.04]} radius={0.02} position={[0.86, -0.36, 0.45]}><meshStandardMaterial color={CLAY.leg} /></RoundedBox>
      <RoundedBox args={[0.26, 0.07, 0.04]} radius={0.02} position={[0.86, -0.5, 0.45]}><meshStandardMaterial color={CLAY.leg} /></RoundedBox>
      {/* legs and antenna */}
      {[-0.75, 0.75].map((x) => (
        <mesh key={x} position={[x, -0.98, 0]} rotation={[0, 0, x > 0 ? -0.18 : 0.18]}><cylinderGeometry args={[0.05, 0.035, 0.4, 16]} /><meshStandardMaterial color={CLAY.leg} /></mesh>
      ))}
      <mesh position={[0.28, 1.12, 0]} rotation={[0, 0, -0.45]}><cylinderGeometry args={[0.025, 0.025, 0.7, 12]} /><meshStandardMaterial color={CLAY.leg} /></mesh>
      <mesh position={[0.43, 1.43, 0]}><sphereGeometry args={[0.07, 24, 24]} /><meshStandardMaterial color={CLAY.pen} /></mesh>
    </group>
  );
}

function Bubble() {
  return (
    <group>
      <RoundedBox args={[1.5, 0.82, 0.26]} radius={0.2} smoothness={6} castShadow><meshStandardMaterial color={CLAY.white} roughness={0.9} /></RoundedBox>
      <mesh position={[-0.5, -0.5, 0]} rotation={[0, 0, 0.6]}><coneGeometry args={[0.13, 0.32, 24]} /><meshStandardMaterial color={CLAY.white} roughness={0.9} /></mesh>
      {/* the same highlighted line as on screen: what the narrator says */}
      <RoundedBox args={[1.04, 0.15, 0.02]} radius={0.04} position={[0, 0.12, 0.14]}><meshStandardMaterial color={CLAY.hl} /></RoundedBox>
      <Line y={0.12} w={0.92} color={CLAY.ink} z={0.16} />
      <Line x={-0.16} y={-0.14} w={0.6} color="#c9bda8" z={0.14} />
    </group>
  );
}

function Pen() {
  const ref = useRef<Group>(null);
  useFrame(({ clock }) => {
    if (ref.current) ref.current.rotation.z = -0.9 + Math.sin(clock.elapsedTime * 0.9) * 0.08;
  });
  return (
    <group ref={ref}>
      <mesh castShadow><cylinderGeometry args={[0.075, 0.075, 1.25, 24]} /><meshStandardMaterial color={CLAY.pen} roughness={0.55} /></mesh>
      <mesh position={[0, 0.66, 0]}><cylinderGeometry args={[0.078, 0.078, 0.12, 24]} /><meshStandardMaterial color={CLAY.ink} roughness={0.4} /></mesh>
      <mesh position={[0, -0.73, 0]} rotation={[Math.PI, 0, 0]}><coneGeometry args={[0.075, 0.2, 24]} /><meshStandardMaterial color={CLAY.white} /></mesh>
      <mesh position={[0, -0.86, 0]} rotation={[Math.PI, 0, 0]}><coneGeometry args={[0.025, 0.08, 16]} /><meshStandardMaterial color={CLAY.ink} /></mesh>
    </group>
  );
}

/** Pull the camera back on narrow (portrait) canvases so the bubble and pen
    stay in frame on phones. */
function FitCamera() {
  const { camera, size } = useThree();
  useEffect(() => {
    const aspect = size.width / Math.max(1, size.height);
    camera.position.z = aspect < 0.8 ? 10.5 : aspect < 1.15 ? 8.4 : 6.2;
    camera.updateProjectionMatrix();
  }, [camera, size.width, size.height]);
  return null;
}

export default function Scene3D() {
  const still = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  return (
    <Canvas shadows dpr={[1, 2]} camera={{ position: [0.4, 0.5, 6.2], fov: 34 }} gl={{ alpha: true, antialias: true }}
      aria-label="A clay TV showing a slide, a speech bubble repeating the highlighted line, and a red pen">
      <FitCamera />
      <ambientLight intensity={0.75} />
      <directionalLight position={[3, 5, 4]} intensity={1.35} castShadow shadow-mapSize={[1024, 1024]} color="#fff3e0" />
      <directionalLight position={[-4, 2, 2]} intensity={0.35} color="#ffd9c2" />
      <PresentationControls global={false} polar={[-0.15, 0.2]} azimuth={[-0.5, 0.5]} snap>
        <group position={[-0.35, -0.05, 0]} rotation={[0.04, -0.28, 0]}>
          <Float speed={still ? 0 : 1.4} rotationIntensity={0.15} floatIntensity={0.35}><TV /></Float>
          <Float speed={still ? 0 : 2} rotationIntensity={0.25} floatIntensity={0.6}>
            <group position={[1.75, 1.15, 0.5]} rotation={[0, -0.25, 0.04]}><Bubble /></group>
          </Float>
          <Float speed={still ? 0 : 1.7} rotationIntensity={0.3} floatIntensity={0.5}>
            <group position={[1.9, -0.55, 0.9]}><Pen /></group>
          </Float>
        </group>
      </PresentationControls>
      <ContactShadows position={[0, -1.25, 0]} opacity={0.35} scale={9} blur={2.4} far={3} color="#5c401e" />
    </Canvas>
  );
}
