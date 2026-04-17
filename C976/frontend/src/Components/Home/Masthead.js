import React from "react";

function Masthead() {
  return (
    <div className="container-fluid masthead-container">
      <div className="row masthead-content">
        <div className="col-12 text-center">
          <h1 className="masthead-title animate__animated animate__fadeInDown">
            Welcome to <span className="highlight">SignLingua</span>
          </h1>
          <div className="divider animate__animated animate__fadeIn animate__delay-1s" />
          <p className="masthead-subtitle animate__animated animate__fadeIn animate__delay-1s">
            Real-time Translation and Tone Modification in Live Chat using Large Language Models with Multimodal Intelligence and Inclusive Communication Support.
          </p>
          <div className="cta-container animate__animated animate__fadeInUp animate__delay-2s">
            <a className="cta-button" href="/sign-kit/chat">
              Try AI Chat <i className="fas fa-comments ml-2" />
            </a>
            <a className="cta-button-outline ml-3" href="#intro">
              Learn More
            </a>
          </div>
        </div>
      </div>

      {/* Animated background elements */}
      <div className="masthead-shapes">
        <div className="shape shape-1"></div>
        <div className="shape shape-2"></div>
        <div className="shape shape-3"></div>
      </div>

      {/* Scroll indicator */}
      <div className="scroll-indicator animate__animated animate__fadeIn animate__delay-3s">
        <span></span>
      </div>
    </div>
  );
}

export default Masthead;
