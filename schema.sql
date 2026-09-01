
-- Name: movie_reviews; Type: TABLE; Schema: public; Owner: -

CREATE TABLE public.movie_reviews (
    review_id text,
    id text,
    review_text text,
    original_score text,
    sentiment text,
    sentiment_conf double precision,
    parent_asin text,
    title text,
    has_image boolean,
    features text[],
    description text[],
    image_url text,
    tag_is_clearly_positive boolean,
    tag_is_a_clearly_positive_review boolean,
    tag_is_a_positive_review boolean,
    tag_is_a_negative_review boolean,
    tag_clearly_positive boolean,
    tag_positive boolean
);

-- Name: movies; Type: TABLE; Schema: public; Owner: -

CREATE TABLE public.movies (
    id text NOT NULL,
    title text,
    audience_score double precision,
    tomato_meter double precision,
    genre text,
    original_language text,
    director text
);

-- Name: predicate_query_history; Type: TABLE; Schema: public; Owner: -

CREATE TABLE public.predicate_query_history (
    predicate text NOT NULL,
    query_label text NOT NULL,
    first_seen_at timestamp with time zone DEFAULT now()
);

-- Name: predicate_store; Type: TABLE; Schema: public; Owner: -

CREATE TABLE public.predicate_store (
    predicate_idx integer NOT NULL,
    predicate_type text NOT NULL,
    predicate_nl text NOT NULL,
    agent_prompt text NOT NULL
);

-- Name: tag_meta; Type: TABLE; Schema: public; Owner: -

CREATE TABLE public.tag_meta (
    parent_asin text NOT NULL,
    predicate_canon text NOT NULL,
    tag_value boolean,
    confidence double precision,
    settled_tier smallint,
    settled_modality text,
    source_updated_at timestamp with time zone DEFAULT now()
);

-- Name: movies movies_pkey; Type: CONSTRAINT; Schema: public; Owner: -

ALTER TABLE ONLY public.movies
    ADD CONSTRAINT movies_pkey PRIMARY KEY (id);

-- Name: predicate_query_history predicate_query_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -

ALTER TABLE ONLY public.predicate_query_history
    ADD CONSTRAINT predicate_query_history_pkey PRIMARY KEY (predicate, query_label);

-- Name: predicate_store predicate_store_pkey; Type: CONSTRAINT; Schema: public; Owner: -

ALTER TABLE ONLY public.predicate_store
    ADD CONSTRAINT predicate_store_pkey PRIMARY KEY (predicate_idx);

-- Name: tag_meta tag_meta_pkey; Type: CONSTRAINT; Schema: public; Owner: -

ALTER TABLE ONLY public.tag_meta
    ADD CONSTRAINT tag_meta_pkey PRIMARY KEY (parent_asin, predicate_canon);

-- Name: idx_movie_reviews_pasin; Type: INDEX; Schema: public; Owner: -

CREATE UNIQUE INDEX idx_movie_reviews_pasin ON public.movie_reviews USING btree (parent_asin);

-- Name: idx_reviews_movie; Type: INDEX; Schema: public; Owner: -

CREATE INDEX idx_reviews_movie ON public.movie_reviews USING btree (id);

-- Name: idx_reviews_sentiment; Type: INDEX; Schema: public; Owner: -

CREATE INDEX idx_reviews_sentiment ON public.movie_reviews USING btree (id, sentiment);

-- Name: idx_tag_meta_pred; Type: INDEX; Schema: public; Owner: -

CREATE INDEX idx_tag_meta_pred ON public.tag_meta USING btree (predicate_canon);
