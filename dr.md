 - https://github.com/thalesfm/differentiable-renderer 

  - **Vector Autograd如何实现** 暂时没看明白
  ```
   template <typename T, std::size_t N, bool Autograd = false>
  class Vector;
  
  template <typename T, std::size_t N>
  class Vector<T, N> {
  public:
      using iterator = typename std::array<T, N>::iterator;
      using const_iterator = typename std::array<T, N>::const_iterator;
  
      Vector() = default;
  
      explicit Vector(T value)
      {
          m_data.fill(value);
      }
  
      Vector(std::initializer_list<T> init)
      {
          if (init.size() != N)
              throw std::runtime_error(
                  "incorrect number of initializers for `Vector`");
          std::copy(init.begin(), init.end(), begin());
      }
  
      T& operator[](std::size_t pos)
      {
          return m_data[pos];
      }
  
      const T& operator[](std::size_t pos) const
      {
          return m_data[pos];
      }
  
      iterator begin()
      {
          return m_data.begin();
      }
  
      const_iterator begin() const
      {
          return m_data.begin();
      }
  
      iterator end()
      {
          return m_data.end();
      }
  
      const_iterator end() const
      {
          return m_data.end();
      }
  
      constexpr std::size_t size() const
      {
          return N;
      }
  
      Vector& operator+=(const Vector& rhs)
      {
          std::transform(begin(), end(), rhs.begin(), begin(),
              [](const T& x, const T& y) { return x + y; });
          return *this;
      }
  
      Vector& operator-=(const Vector& rhs)
      {
          std::transform(begin(), end(), rhs.begin(), begin(),
              [](T x, T y) { return x - y; });
          return *this;
      }
  
      Vector& operator*=(const Vector& rhs)
      {
          std::transform(begin(), end(), rhs.begin(), begin(),
              [](T x, T y) { return x * y; });
          return *this;
      }
  
      Vector& operator*=(T s)
      {
          std::transform(begin(), end(), begin(),
              [=](T x) { return x * s; });
          return *this;
      }
  
      Vector& operator/=(const Vector& rhs)
      {
          std::transform(begin(), end(), rhs.begin(), begin(),
              [](T x, T y) { return x / y; });
          return *this;
      }
  
      Vector& operator/=(T s)
      {
          std::transform(begin(), end(), begin(),
              [=](T x) { return x / s; });
          return *this;
      }
  
  private:
      std::array<T, N> m_data;
  };
  
  namespace internal {
  
  template <typename T, std::size_t N>
  class AutogradNode : public Vector<T, N> {
  public:
      using Vector<T, N>::Vector;
  
      AutogradNode(const Vector<T, N>& v)
        : Vector<T, N>(v)
      { }
  
      virtual ~AutogradNode()
      { }
  
      virtual Vector<T, N>& grad()
      {
          throw std::runtime_error("Vector has no gradient (not a variable)");
      }
  
      virtual const Vector<T, N>& grad() const
      {
          throw std::runtime_error("Vector has no gradient (not a variable)");
      }
  
      virtual bool requires_grad() const = 0;
  
      virtual void backward(const Vector<T, N>& grad) const = 0;
  };
  
  template <typename T, std::size_t N>
  class ConstantNode : public AutogradNode<T, N> {
  public:
      ConstantNode(const Vector<T, N>& v)
        : AutogradNode<T, N>(v)
      { }
  
      bool requires_grad() const override
      {
          return false;
      }
  
      void backward(const Vector<T, N>& grad) const override
      { }
  };
  
  template <typename T, std::size_t N>
  class VariableNode : public AutogradNode<T, N> {
  public:
      using AutogradNode<T, N>::AutogradNode;
  
      Vector<T, N>& grad() override
      {
          return m_grad;
      }
  
      const Vector<T, N>& grad() const override
      {
          return m_grad;
      }
  
      bool requires_grad() const override
      {
          return true;
      }
  
      void backward(const Vector<T, N>& grad) const override
      {
          m_grad += grad;
      }
  
  private:
      mutable Vector<T, N> m_grad;
  };
  
  template <typename T, std::size_t N, typename Backward>
  class BackwardNode : public AutogradNode<T, N> {
  public:
      BackwardNode(const Vector<T, N>& v, const Backward& backward)
        : AutogradNode<T, N>(v), m_backward(backward)
      { }
  
      bool requires_grad() const override
      {
          return true;
      }
  
      void backward(const Vector<T, N>& grad) const override
      {
          m_backward(grad);
      }
  
  private:
      typename std::decay_t<Backward> m_backward;
  };
  
  } // namespace internal
  
  template <typename T, std::size_t N>
  class Vector<T, N, true> {
  public:
      explicit Vector(T value, bool requires_grad = false)
        : Vector(Vector<T, N>(value), requires_grad)
      { }
  
      Vector(std::initializer_list<T> init, bool requires_grad = false)
        : Vector(Vector<T, N>(init), requires_grad)
      { }
  
      Vector(const Vector<T, N>& v, bool requires_grad = false)
      {
          if (requires_grad)
              m_ptr = std::make_shared<internal::VariableNode<T, N>>(v);
          else
              m_ptr = std::make_shared<internal::ConstantNode<T, N>>(v);
      }
  
      template <typename Backward>
      Vector(const Vector<T, N>& v, const Backward& backward)
        : m_ptr(new internal::BackwardNode<T, N, Backward>(v, backward))
      { }
  
      T& operator[](std::size_t pos)
      {
          return (*m_ptr)[pos];
      }
  
      const T& operator[](std::size_t pos) const
      {
          return (*m_ptr)[pos];
      }
  
      constexpr std::size_t size() const
      {
          return N;
      }
  
      Vector<T, N>& detach()
      {
          return *m_ptr;
      }
  
      const Vector<T, N>& detach() const
      {
          return *m_ptr;
      }
  
      Vector<T, N>& grad()
      {
          return m_ptr->grad();
      }
  
      const Vector<T, N>& grad() const
      {
          return m_ptr->grad();
      }
  
      bool requires_grad() const
      {
          return m_ptr->requires_grad();
      }
  
      void backward(const Vector<T, N>& grad) const
      {
          m_ptr->backward(grad);
      }
  
      Vector<T, N, true>& operator+=(const Vector<T, N, true>& rhs)
      {
          return *this = *this + rhs;
      }
  
      Vector<T, N, true>& operator-=(const Vector<T, N, true>& rhs)
      {
          return *this = *this - rhs;
      }
  
      Vector<T, N, true>& operator*=(const Vector<T, N, true>& rhs)
      {
          return *this = *this * rhs;
      }
  
      Vector<T, N, true>& operator*=(T s)
      {
          return *this = *this * s;
      }
  
      Vector<T, N, true>& operator/=(const Vector<T, N, true>& rhs)
      {
          return *this = *this / rhs;
      }
  
      Vector<T, N, true>& operator/=(T s)
      {
          return *this = *this / s;
      }
  
  private:
      std::shared_ptr<internal::AutogradNode<T, N>> m_ptr;
  };
  
  template <typename T, std::size_t N, bool Ag,
            typename = typename std::enable_if_t<std::is_convertible_v<int, T>>>
  inline Vector<T, N, Ag> operator-(const Vector<T, N, Ag>& v)
  {
      return -1 * v;
  }
  
  template <typename T, std::size_t N>
  inline Vector<T, N> operator+(Vector<T, N> lhs, const Vector<T, N>& rhs)
  {
      return lhs += rhs;
  }
  
  template <typename T, std::size_t N>
  inline Vector<T, N> operator-(Vector<T, N> lhs, const Vector<T, N>& rhs)
  {
      return lhs -= rhs;
  }
  
  template <typename T, std::size_t N>
  inline Vector<T, N> operator*(Vector<T, N> lhs, const Vector<T, N>& rhs)
  {
      return lhs *= rhs;
  }
  
  template <typename T, std::size_t N, typename S,
            typename = typename std::enable_if_t<std::is_convertible_v<S, T>>>
  inline Vector<T, N> operator*(Vector<T, N> v, S s)
  {
      return v *= s;
  }
  
  template <typename T, std::size_t N, typename S,
            typename = typename std::enable_if_t<std::is_convertible_v<S, T>>>
  inline Vector<T, N> operator*(S s, Vector<T, N> v)
  {
      return v *= s;
  }
  
  template <typename T, std::size_t N>
  inline Vector<T, N> operator/(Vector<T, N> lhs, const Vector<T, N>& rhs)
  {
      return lhs /= rhs;
  }
  
  template <typename T, std::size_t N, typename S,
            typename = typename std::enable_if_t<std::is_convertible_v<S, T>>>
  inline Vector<T, N> operator/(Vector<T, N> v, S s)
  {
      return v /= s;
  }
  
  template <typename T, std::size_t N>
  inline Vector<T, N>& detach(Vector<T, N>& v)
  {
      return v;
  }
  
  template <typename T, std::size_t N>
  inline const Vector<T, N>& detach(const Vector<T, N>& v)
  {
      return v;
  }
  
  template <typename T, std::size_t N>
  inline Vector<T, N>& detach(Vector<T, N, true>& v)
  {
      return v.detach();
  }
  
  template <typename T, std::size_t N>
  inline const Vector<T, N>& detach(const Vector<T, N, true>& v)
  {
      return v.detach();
  }
  
  template <typename T, std::size_t N>
  inline constexpr bool requires_grad(const Vector<T, N>& v)
  {
      return false;
  }
  
  template <typename T, std::size_t N>
  inline bool requires_grad(const Vector<T, N, true>& v)
  {
      return v.requires_grad();
  }
  
  template <typename T, std::size_t N>
  inline void backward(Vector<T, N>& v, const Vector<T, N>& grad)
  { }
  
  template <typename T, std::size_t N>
  inline void backward(Vector<T, N, true>& v, const Vector<T, N>& grad)
  {
      v.backward(grad);
  }
  
  namespace internal {
  
  template <typename T, std::size_t N>
  struct AddBackward {
      void operator()(const Vector<T, N>& grad) const
      {
          lhs.backward(grad);
          rhs.backward(grad);
      }
  
      Vector<T, N, true> lhs, rhs;
  };
  
  template <typename T, std::size_t N>
  struct SubBackward {
      void operator()(const Vector<T, N>& grad) const
      {
          lhs.backward(grad);
          rhs.backward(-grad);
      }
  
      Vector<T, N, true> lhs, rhs;
  };
  
  template <typename T, std::size_t N>
  struct MulBackward {
      void operator()(const Vector<T, N>& grad) const
      {
          lhs.backward(rhs.detach() * grad);
          rhs.backward(lhs.detach() * grad);
      }
  
      Vector<T, N, true> lhs, rhs;
  };
  
  template <typename T, std::size_t N>
  struct ScalarMulBackward {
      void operator()(const Vector<T, N>& grad) const
      {
          v.backward(s * grad);
      }
  
      T s;
      Vector<T, N, true> v;
  };
  
  template <typename T, std::size_t N>
  struct DivBackward {
      void operator()(const Vector<T, N>& grad) const
      {
          lhs.backward(grad / rhs.detach());
          rhs.backward(-lhs.detach() * grad / (rhs.detach() * rhs.detach()));
      }
  
      Vector<T, N, true> lhs, rhs;
  };
  
  template <typename T, std::size_t N>
  struct ScalarDivBackward {
      void operator()(const Vector<T, N>& grad) const
      {
          v.backward(grad / s);
      }
  
      Vector<T, N, true> v;
      T s;
  };
  
  } // namespace internal
  
  template <typename T, std::size_t N, bool Ag1, bool Ag2,
            typename = typename std::enable_if_t<Ag1 || Ag2>>
  inline Vector<T, N, true> operator+(const Vector<T, N, Ag1>& lhs,
                                      const Vector<T, N, Ag2>& rhs)
  {
      auto r = detach(lhs) + detach(rhs);
      if (!requires_grad(lhs) && !requires_grad(rhs))
          return r;
      return Vector<T, N, true>(r, internal::AddBackward<T, N>{lhs, rhs});
  }
  
  template <typename T, std::size_t N, bool Ag1, bool Ag2,
            typename = typename std::enable_if_t<Ag1 || Ag2>>
  inline Vector<T, N, true> operator-(const Vector<T, N, Ag1>& lhs,
                                      const Vector<T, N, Ag2>& rhs)
  {
      auto r = detach(lhs) - detach(rhs);
      if (!requires_grad(lhs) && !requires_grad(rhs))
          return r;
      return Vector<T, N, true>(r, internal::SubBackward<T, N>{lhs, rhs});
  }
  
  template <typename T, std::size_t N, bool Ag1, bool Ag2,
            typename = typename std::enable_if_t<Ag1 || Ag2>>
  inline Vector<T, N, true> operator*(const Vector<T, N, Ag1>& lhs,
                                      const Vector<T, N, Ag2>& rhs)
  {
      auto r = detach(lhs) * detach(rhs);
      if (!requires_grad(lhs) && !requires_grad(rhs))
          return r;
      return Vector<T, N, true>(r, internal::MulBackward<T, N>{lhs, rhs});
  }
  
  template <typename T, std::size_t N, typename S,
            typename = typename std::enable_if_t<std::is_convertible_v<S, T>>>
  inline Vector<T, N, true> operator*(Vector<T, N, true> v, S s)
  {
      return s * v;
  }
  
  template <typename T, std::size_t N, typename S,
            typename = typename std::enable_if_t<std::is_convertible_v<S, T>>>
  inline Vector<T, N, true> operator*(S s, Vector<T, N, true> v)
  {
      auto r = s * v.detach();
      if (!v.requires_grad())
          return r;
      return Vector<T, N, true>(r, internal::ScalarMulBackward<T, N>{s, v});
  }
  
  template <typename T, std::size_t N, bool Ag1, bool Ag2,
            typename = typename std::enable_if_t<Ag1 || Ag2>>
  inline Vector<T, N, true> operator/(const Vector<T, N, Ag1>& lhs,
                                      const Vector<T, N, Ag2>& rhs)
  {
      auto r = detach(lhs) / detach(rhs);
      if (!requires_grad(lhs) && !requires_grad(rhs))
          return r;
      return Vector<T, N, true>(r, internal::DivBackward<T, N>{lhs, rhs});
  }
  
  template <typename T, std::size_t N, typename S,
            typename = typename std::enable_if_t<std::is_convertible_v<S, T>>>
  inline Vector<T, N, true> operator/(Vector<T, N, true> v, S s)
  {
      auto r = v.detach() / s;
      if (!v.requires_grad())
          return r;
      return Vector<T, N, true>(r, internal::ScalarDivBackward<T, N>{v, s});
  }
  
  template <typename T, std::size_t N, bool Ag>
  inline std::ostream& operator<<(std::ostream& os, const Vector<T, N, Ag>& v)
  {
      os << "Vector<" << typeid(T).name() << ", " << N;
      if (Ag)
          os << ", true";
      os << ">{";
      for (std::size_t i = 0; i < v.size()-1; ++i)
          os << v[i] << ", ";
      if (v.size() > 0)
          os << v[v.size()-1];
      return os << "}";
  }
  
  template <typename T, std::size_t N>
  inline T dot(const Vector<T, N>& lhs, const Vector<T, N>& rhs)
  {
      Vector<T, N> tmp = lhs * rhs;
      return std::accumulate(tmp.begin(), tmp.end(), T());
  }
  
  template <typename T, std::size_t N>
  inline T norm(const Vector<T, N>& v)
  {
      return sqrt(dot(v, v));
  }
  
  template <typename T, std::size_t N>
  inline Vector<T, N> normalize(const Vector<T, N>& v)
  {
      return v / norm(v);
  }
  
  template <typename T>
  inline Vector<T, 3> cross(const Vector<T, 3>& lhs, const Vector<T, 3>& rhs)
  {
      Vector<T, 3> r;
      r[0] = lhs[1]*rhs[2] - lhs[2]*rhs[1];
      r[1] = lhs[2]*rhs[0] - lhs[0]*rhs[2];
      r[2] = lhs[0]*rhs[1] - lhs[1]*rhs[0];
      return r;
  }
  
  template <typename T, std::size_t N>
  inline Vector<T, N> reflect(const Vector<T, N>& v, const Vector<T, N>& n)
  {
      return -v + 2*dot(n, v)*n;
  }
  ```
  

  - Emitter
  ```
  template<typename T> class Emitter{
    virtual Vector<T, 3, true> emission() const = 0; // 返回的是啥
  };

  template<typename T>
  class AreaEmitter : public Emitter<T>{
    AreaEmitter(Vector<T, 3, true> emission);

    Vector<T, 3, true> emission() const override{ return m_emission; } // 一个常向量， 代表啥呢？
  }；
  ```
  

  - BxDF
  ```
  class BxDF{

    virtual Vector<T, 3, true> operator()(const Vector<T, 3>& normal, const Vector<T, 3>& dir_in, const Vector<T, 3>& dir_out) const = 0;

    virtual std::tuple<Vector<T, 3>, double> sample(const Vector<T, 3>& normal, const Vector<T, 3>& dir_in) const = 0; // return the output direction + pdf value.
  };

  std::array<Vector<T, 3>, 3> make_frame(const Vector<T, 3>& normal)
  {
    e1 = {1, 0, 0};
    e2 = {0, 1, 0};

    tangent = ...;
    bitangent = ...;

    return {tangent, bitangent, normal};
  }

  Vector<T, 3> angle_to_dir(theta, phi, const std::array<Vector<T, 3>, 3>& frame)
  {
    x = cos(phi) * sin(theta);  // it seems phi's range is [0, 360]
    y = sin(phi) * sin(theta);
    z = cos(theta);             // it seems the's range is [0, 180] 维度方向从北极到南极？

    return x * frame[0] + y * frame[1] + z * frame[2];
  }
  


  class DiffuseBxDF : public BxDF<T>
  {
  Vector<T, 3, true> operator()(normal, dir_in, dir_out) override{ return m_color/pi; }
  
  std::tuple<Vector<T, 3>, double> sample(const Vector<T, 3>& normal, const Vector<T, 3>& dir_in) const{
    // diffuse model's out_dir is not related to dir_in.

    // out_dir is randomly sampled
    theta = asin(random::uniform());
    phi = 2 * pi * random::uniform();
    frame = make_frame(normal);
    dir = angle_to_dir(theta, phi, frame);
    pdf = cos(theta)/pi;                         // why this is cos(theta)/pi
    reutrn std::make_tuple(dir, pdf);
  }
  Vector<T, 3, true> m_color;
  };

  template <typename T>
  class SpecularBxDF : public BxDF<T> {
  public:
      SpecularBxDF(const Vector<T, 3, true>& color, double exponent)
        : m_color(color)
        , m_exponent(exponent)
      { }
  
      Vector<T, 3, true> operator()(
          const Vector<T, 3>& normal,
          const Vector<T, 3>& dir_in,
          const Vector<T, 3>& dir_out) const override
      {
          Vector<T, 3> halfway = normalize(dir_in + dir_out);
          double cos_theta = dot(normal, halfway);
          double sin_theta = sqrt(1 - cos_theta*cos_theta);
          double factor = (m_exponent + 2) / (2 * pi)
              * pow(cos_theta, m_exponent) * sin_theta;
          return factor * m_color;
      }
  
      std::tuple<Vector<T, 3>, double> sample(
          const Vector<T, 3>& normal,
          const Vector<T, 3>& dir_in) const override
      {
          double theta = acos(sqrt(pow(random::uniform(), 2/(m_exponent+2))));
          double phi = 2 * pi * random::uniform();
          auto frame = internal::make_frame(normal);
          auto halfway = internal::angle_to_dir(theta, phi, frame);
          if (dot(halfway, dir_in) < 0)
              halfway = reflect(halfway, normal);
          auto dir = reflect(dir_in, halfway);
          double pdf = (m_exponent + 2) / (2 * pi) *
              pow(cos(theta), m_exponent+1) * sin(theta);
          return std::make_tuple(dir, pdf);
      }
  private:
      Vector<T, 3, true> m_color;
      double m_exponent;
  };
  
  template <typename T>
  class MirrorBxDF : public BxDF<T> {
  public:
      Vector<T, 3, true> operator()(
          const Vector<T, 3>& normal,
          const Vector<T, 3>& dir_in,
          const Vector<T, 3>& dir_out) const override
      {
          double cos_theta = dot(normal, dir_out);
          return 1 / cos_theta;
      }
  
      std::tuple<Vector<T, 3>, double> sample(
          const Vector<T, 3>& normal,
          const Vector<T, 3>& dir_in) const override
      {
          return std::make_tuple(reflect(dir_in, normal), 1);
      }
  };
  
  ```
  
  - Light
  ```
  用Shape表示， 点光源， Sphere<T> light(center, radius, nullptr/*no BxDF*/, emitter);
  ```

  - scene
  ```
  
  template<typename T>
  class Shape {
  Shape(BxDF<T>* bxdf = nullptr, Emitter<T> emitter = nullptr);
  
  virtual bool intersect(Vector<T, 3> orig, Vector<T, 3> dir, double& t) const = 0;

  virtual Vector<T, 3> normal(Vector<T, 3> point) const = 0;

  BxDF<T>* bxdf();
  Emitter<T> emitter();
  };
  
  
  template<typename T>
  using Scene = std::vector<Shape<T>*>;
  ```


  - camera
  ```
  template<typename T>
  class Camera {
    Camera(width, height, vfov, eye_position = (0, 0, 0), forward = (0, 0, -1), right = (1, 0, 0), up = (0, 1, 0));

    void look_at(eye, at, up = (0, 1, 0));  // Setup m_eye, m_forward, m_right, m_up

    int width() const;
    int height() const;
    Vector<T, 3> eye() const;
    
    double aspect() const{return double(m_width)/m_height; }
  
    // ray tracing related sample ray
    std::tuple<Vector<T, 3>, double> sample(x, y) const{
      // 1. y is from top to down, that's why * -m_up
      // 2. fov is in the field of view angle in the y direction, that's why m_right need multiply aspect() which is width/height
      double s = (x + random::uniform()) / m_width; // [0.0, 1.0]
      double t = (y + random::uniform()) / m_height; // [0.0, 1.0]
      Vector<T, 3> dir = m_forward;
      dir += (2*s - 1) * aspect() * tan(vfov/2) * m_right;
      dir += (2*t - 1) * tan(vfov/2) * -m_up;
      dir = normalize(dir);
      return std::make_tuple(dir, 1);
    }
  };
  ```

  - Integrator
  ```
  template <typename T, std::size_t N, typename Forward, typename Sampler>
  struct IntegrateBackward {
      void operator()(const Vector<T, N>& grad) const
      {
          for (std::size_t i = 0; i < n_samples; ++i) {
              auto [sample, pdf] = sampler();
              forward(sample).backward(grad / pdf);
          }
      }
  
      typename std::decay<Forward>::type forward;
      typename std::decay<Sampler>::type sampler;
      std::size_t n_samples;
  };
  
  template <typename T, std::size_t N, typename Forward, typename Sampler>
  inline Vector<T, N, true> integrate_biased(const Forward& forward,
                                             const Sampler& sampler,
                                             std::size_t n_samples)
  {
      Vector<T, N, true> r(0);
      for (std::size_t i = 0; i < n_samples; ++i) {
          auto [sample, pdf] = sampler();
          r += forward(sample) / pdf;
      }
      return r;
  }
  
  template <typename T, std::size_t N, typename Forward, typename Sampler>
  inline Vector<T, N, true> integrate_unbiased(const Forward& forward,
                                               const Sampler& sampler,
                                               std::size_t n_samples)
  {
      Vector<T, N> r(0);
      for (std::size_t i = 0; i < n_samples; ++i) {
          auto [sample, pdf] = sampler();
          r += forward(sample).detach() / pdf;
      }
      return Vector<T, N, true>(r,
          IntegrateBackward<T, N, Forward, Sampler>
              {forward, sampler, n_samples});
  }
  
  } // namespace internal
  
  template <typename T, std::size_t N, typename Forward, typename Sampler>
  inline Vector<T, N, true> integrate(const Forward& forward,
                                      const Sampler& sampler,
                                      std::size_t n_samples,
                                      bool unbiased = false)
  {
      if (unbiased)
          return internal::integrate_unbiased<T, N>(forward, sampler, n_samples);
      else
          return internal::integrate_biased<T, N>(forward, sampler, n_samples);
  }
  ```
  
  - PathTracer
  ```
  template <typename T>
  using Scene = std::vector<Shape<T>*>;
  
  namespace internal {
  
  template <typename T>
  std::tuple<Vector<T, 3>, double> sample_bxdf(
      const BxDF<T> *bxdf,
      Vector<T, 3> normal,
      Vector<T, 3> dir_in)
  {
      if (bxdf)
          return bxdf->sample(normal, dir_in);
      else
          return std::make_tuple(Vector<T, 3>(0), 1);
  }
  
  template <typename T>
  Vector<T, 3, true> eval_bxdf(
      const BxDF<T> *bxdf,
      Vector<T, 3> normal,
      Vector<T, 3> dir_in,
      Vector<T, 3> dir_out)
  {
      if (bxdf)
          return (*bxdf)(normal, dir_in, dir_out);
      else
          return Vector<T, 3>(0);
  }
  
  template <typename T>
  Vector<T, 3, true> emission(const Emitter<T> *emitter)
  {
      if (emitter)
          return emitter->emission();
      else
          return Vector<T, 3>(0);
  }
  
  } // namespace internal
  
  template <typename T>
  class Pathtracer {
  public:
      Pathtracer(double absorb, std::size_t min_bounces)
        : m_absorb(absorb), m_min_bounces(min_bounces) { }
  
      Vector<T, 3, true> trace(const Scene<T>& scene,
                               Vector<T, 3> orig,
                               Vector<T, 3> dir,
                               std::size_t depth = 0) const;
  
  private:
      struct RaycastHit {
          Vector<T, 3> point;
          Vector<T, 3> normal;
          BxDF<T> *bxdf;
          Emitter<T> *emitter;
      };
  
      bool raycast(const Scene<T>& scene,
                   Vector<T, 3> orig,
                   Vector<T, 3> dir,
                   RaycastHit& hit) const
      {
          double tmin = inf;
          for (auto shape : scene) {
              double t;
              if (!shape->intersect(orig, dir, t) || t >= tmin)
                  continue;
              tmin = t;
              hit.point = orig + t*dir;
              hit.normal = shape->normal(hit.point);
              hit.bxdf = shape->bxdf();
              hit.emitter = shape->emitter();
          }
          return !std::isinf(tmin);
      }
  
      Vector<T, 3, true> scatter(const Scene<T>& scene,
                                 RaycastHit& hit,
                                 Vector<T, 3> dir_in,
                                 std::size_t depth) const
      {
          Vector<T, 3, true> diffuse = integrate<T, 3>(
              [=](const Vector<T, 3>& dir_out)
              {
                  Vector<T, 3> orig = hit.point + 1e-3*dir_out;
                  Vector<T, 3, true> brdf_value = internal::eval_bxdf(
                      hit.bxdf, hit.normal, -dir_in, dir_out);
                  Vector<T, 3, true> radiance = trace(scene, orig, dir_out, depth+1);
                  double cos_theta = dot(hit.normal, dir_out);
                  return brdf_value * radiance * cos_theta;
              },
              [=]()
              {
                  return internal::sample_bxdf(hit.bxdf, hit.normal, -dir_in);
              },
              1,
              false
          );
          Vector<T, 3, true> emission = internal::emission(hit.emitter);
          return emission + diffuse;
      }
  
      double m_absorb;
      std::size_t m_min_bounces;
  };
  
  template <typename T>
  Vector<T, 3, true> Pathtracer<T>::trace(const Scene<T>& scene,
                                          Vector<T, 3> orig,
                                          Vector<T, 3> dir,
                                          std::size_t depth) const
  
  {
      if (depth >= m_min_bounces && random::uniform() < m_absorb)
          return Vector<T, 3>(0);
      double p = depth >= m_min_bounces ? (1 - m_absorb) : 1;
      RaycastHit hit;
      if (raycast(scene, orig, dir, hit))
          return scatter(scene, hit, dir, depth) / p;
      else
          return Vector<T, 3>(0);
  }

  ```

  - total framework
  ```
  using T = double;
  // using T = Dual<double>;

  // Configure scene parameters
  Vector<T, 3, true> red(Vector<T, 3>{0.5, 0, 0}, true);
  Vector<T, 3, true> green(Vector<T, 3>{0, 0.5, 0}, true);
  Vector<T, 3, true> white(Vector<T, 3>{0.5, 0.5, 0.5}, true);
  Vector<T, 3, true> emission(Vector<T, 3>(1), true);

  // Configure scene materials
  auto diffuse_red = std::make_shared<DiffuseBxDF<T>>(red);
  auto diffuse_green = std::make_shared<DiffuseBxDF<T>>(green);
  auto diffuse_white = std::make_shared<DiffuseBxDF<T>>(white);
  auto specular_white = std::make_shared<SpecularBxDF<T>>(white, 30);
  auto emitter = std::make_shared<AreaEmitter<T>>(emission);

  // Configure scene shapes
  Sphere<T> sphere_front(Vector<T, 3>{0., 0., 3.}, 1., diffuse_white);
  Sphere<T> sphere_back(Vector<T, 3>{-1., 1., 4.5}, 1., diffuse_white);
  Plane<T> left_plane(Vector<T, 3>{-1., 0., 0.}, -3., diffuse_red);
  Plane<T> right_plane(Vector<T, 3>{1., 0., 0.1}, -3., diffuse_green);
  Plane<T> back_plane(Vector<T, 3>{0., 0., -1.}, -6., diffuse_white);
  Plane<T> front_plane(Vector<T, 3>{0, 0, 1}, 0, diffuse_white);
  Plane<T> ground_plane(Vector<T, 3>{0., 1., 0.}, -3., diffuse_white);
  Plane<T> ceiling_plane(Vector<T, 3>{0., -1., 0.}, -3., diffuse_white);
  Sphere<T> light(Vector<T, 3>{0., 3., 3.}, 1., nullptr, emitter);

  // Build test scene
  Scene<T> scene;
  scene.push_back(&sphere_front);
  scene.push_back(&sphere_back);
  scene.push_back(&left_plane);
  scene.push_back(&right_plane);
  scene.push_back(&back_plane);
  scene.push_back(&front_plane);
  scene.push_back(&ground_plane);
  scene.push_back(&ceiling_plane);
  scene.push_back(&light);

  // Configure camera position and resolution
  std::size_t width = args.width;
  std::size_t height = args.height;
  Camera<T> cam(width, height);
  cam.look_at(Vector<T, 3>{0, 0, 0}, Vector<T, 3>{0, 0, 1});
  Vector<double, 3> *img = new Vector<double, 3>[width * height];

  // Configure path tracer sampling
  Pathtracer<T> tracer(args.absorb_prob, args.min_bounces);

  // Render test scene
  for (std::size_t y = 0; y < cam.height(); ++y) {
      for (std::size_t x = 0; x < cam.width(); ++x) {
          Vector<T, 3> pixel_radiance(0);
          for (std::size_t i = 0; i < args.samples; ++i) {
              auto [dir, pdf] = cam.sample(x, y);
              Vector<T, 3, true> radiance = tracer.trace(scene, cam.eye(), dir);
              pixel_radiance += radiance.detach() / pdf;
          // Uncomment to compute gradients
              //radiance.backward(Vector<double, 3>(1));
          }
          img[y*width + x] = pixel_radiance / args.samples;
      }
      printf("% 5.2f%%\r", 100. * (y+1) / cam.height());
      fflush(stdout);
  }
  printf("\n");

  // Write radiance to file
  write_exr(args.output.c_str(), img, width, height);

  ```


  - write exr file
  ```
  #include <ImfRgba.h>
  #include <ImfRgbaFile.h>
  Imf::RgbaOutputFile file(filename, width, height, Imf::WRITE_RGBA);
  file.setFrameBuffer(pixels.dasta(), 1, width);
  file.writePixels(height);
  ```
